"""Ascend runtime profiles resolve defaults and isolate lifecycle changes."""
import asyncio
from dataclasses import replace
import json

import httpx
import pytest

from admission_support import backend
from moqe_serving.config import Launch, Node, Replica, Settings
from moqe_serving.deployment.config import NodePreparation, RuntimeProfile
from moqe_serving.deployment.prepare import NodePreparer
from moqe_serving.pool.controller import ConfigController
from moqe_serving.pool.registry import Registry


def replica(name="one", profile="int4", device=0):
    return Replica(name, "awq", f"http://{name}/v1", "backend", node_id="node",
        device_ids=(device,), model_path="/models/awq", runtime_profile=profile,
        launch=Launch(("python3", "{backend_helper}", "start", "--", "python3", "-m",
                       "{backend_module}", "{vllm_args}"),
                      ("python3", "{backend_helper}", "stop")))


def test_defaults_overrides_and_grouped_ascend_checks():
    profile = RuntimeProfile(adapter="moqe_ascend_int4", env={"OMP_NUM_THREADS": "4"},
                             vllm_args={"max_num_seqs": 8})
    node = Node("node", "remote", prepare=NodePreparation(env={"OMP_NUM_THREADS": "6"}))
    one = replica()
    one = replace(one, launch=replace(one.launch, env={"OMP_NUM_THREADS": "2"},
                                     vllm_args={"max_num_seqs": 16}))
    settings = Settings(nodes=(node,), replicas=(one, replica("two", device=1)),
                        runtime_profiles={"int4": profile})
    effective = settings.effective_replica(one)
    assert effective.launch.env["OMP_NUM_THREADS"] == "2"
    assert effective.launch.env["ASCEND_RT_VISIBLE_DEVICES"] == "{device_ids}"
    assert effective.launch.vllm_args == {"quantization": "moqe_ascend_int4", "max_num_seqs": 16}
    assert one.launch.vllm_args == {"max_num_seqs": 16}  # Source config remains unchanged.
    groups = settings.preparation_groups(node, settings.replicas)
    assert len(groups) == 1
    options = groups[0][1].prepare
    assert options.env["ASCEND_RT_VISIBLE_DEVICES"] == "0,1"
    assert options.required_modules == ("vllm", "torch_npu", "vllm_ascend")
    assert options.host_checks == (("npu-smi", "info"),)


def test_native_adapter_leaves_quantization_to_replica():
    settings = Settings(nodes=(Node("node", "remote"),), replicas=(replica(profile="native"),),
                        runtime_profiles={"native": RuntimeProfile()})
    assert settings.effective_replica(settings.replicas[0]).launch.vllm_args == {}
    assert settings.preparation_groups(settings.nodes[0], settings.replicas)[0][1].prepare


@pytest.mark.parametrize("kwargs", [
    {"accelerator": "cuda"}, {"backend": "other"}, {"device_env": "CUDA_VISIBLE_DEVICES"},
    {"environment_scripts": ("relative.sh",)}, {"vllm_args": {"max_num_seqs": 0}},
])
def test_invalid_profile_is_rejected(kwargs):
    with pytest.raises(ValueError):
        RuntimeProfile(**kwargs)


def test_unknown_reference_and_conflicting_cli_fail_before_lifecycle():
    node = Node("node", "remote")
    with pytest.raises(ValueError, match="Unknown runtime profile"):
        Settings(nodes=(node,), replicas=(replica(),))
    conflicting = replace(replica(), launch=Launch(
        ("start", "--quantization", "awq", "{vllm_args}"), ("stop",)))
    with pytest.raises(ValueError, match="both"):
        Settings(nodes=(node,), replicas=(conflicting,),
                 runtime_profiles={"int4": RuntimeProfile(adapter="moqe_ascend_int4")})


def test_profile_edit_drains_only_affected_instances_and_preserves_other_pool_capacity(monkeypatch):
    async def scenario():
        events, preparation = [], []
        async def prepare(self, node, members, **kwargs):
            preparation.append(members[0].runtime_profile)
            return {"passed": True}
        monkeypatch.setattr(NodePreparer, "prepare", prepare)
        profiles = {"int4": RuntimeProfile(env={"ADAPTER_VERSION": "old"}), "other": RuntimeProfile()}
        settings = Settings(nodes=(Node("node", "remote"),),
            replicas=(replica(), replica("two", "other", 1)), runtime_profiles=profiles)
        registry = Registry(())
        async with httpx.AsyncClient(transport=httpx.MockTransport(backend)) as client:
            controller = ConfigController(settings, registry, client)
            async def command(current, action, config):
                events.append((action, current.id, current.launch.env.get("ADAPTER_VERSION")))
            controller.command = command
            await controller.reconcile(settings)
            lease = registry.acquire("awq")
            assert lease.id == "one"
            desired = replace(settings, runtime_profiles={**profiles,
                "int4": replace(profiles["int4"], env={"ADAPTER_VERSION": "new"})})
            task = asyncio.create_task(controller.reconcile(desired))
            await asyncio.sleep(0.01)
            assert registry.states == {"one": "draining", "two": "ready"}
            registry.release(lease)
            await task
            assert events == [("start", "one", "old"), ("start", "two", None),
                              ("stop", "one", "old"), ("start", "one", "new")]
            assert preparation == ["int4", "other", "int4"]
            before = list(events)
            await controller.reconcile(replace(desired, runtime_profiles={**desired.runtime_profiles,
                                                                         "unused": RuntimeProfile()}))
            assert events == before
    asyncio.run(scenario())


def test_failed_adapter_does_not_block_another_profile_on_same_node(monkeypatch):
    async def scenario():
        async def prepare(self, node, members, **kwargs):
            return {"passed": members[0].runtime_profile != "broken", "failed_check": "runtime", "error": "import failed"}
        monkeypatch.setattr(NodePreparer, "prepare", prepare)
        settings = Settings(nodes=(Node("node", "remote"),),
            replicas=(replica(profile="broken"), replica("two", "healthy", 1)),
            runtime_profiles={"broken": RuntimeProfile(), "healthy": RuntimeProfile()})
        registry, launched = Registry(()), []
        async with httpx.AsyncClient(transport=httpx.MockTransport(backend)) as client:
            controller = ConfigController(settings, registry, client)
            async def command(current, action, config):
                launched.append(current.id)
            controller.command = command
            await controller.reconcile(settings)
            assert registry.states == {"one": "unhealthy", "two": "ready"}
            assert launched == ["two"]
            report = controller.snapshot()["nodes"][0]["preparation"]
            assert not report["passed"] and report["profiles"]["healthy"]["passed"]
    asyncio.run(scenario())


def test_profile_load_and_raw_launch_command_resolve_at_execution(tmp_path, monkeypatch):
    async def scenario():
        value = {"runtime_profiles": {"int4": {"adapter": "moqe_ascend_int4", "env": {"FLAG": "enabled"}}},
                 "nodes": [{"id": "node", "host": "remote"}], "replicas": [{
            "id": "one", "expert": "awq", "model": "backend", "node_id": "node",
            "runtime_profile": "int4", "device_ids": [3], "model_path": "/models/awq",
            "base_url": "http://good/v1", "launch": {
                "start_command": ["python3", "{backend_helper}", "start", "--", "python3", "-m",
                                  "{backend_module}", "{vllm_args}"], "stop_command": ["stop"]}}]}
        path = tmp_path / "cluster.json"
        path.write_text(json.dumps(value))
        settings = Settings.load(path)
        calls = []
        async def run(node, command, **kwargs):
            calls.append((command, kwargs))
        monkeypatch.setattr("moqe_serving.pool.controller.run_node_command", run)
        async with httpx.AsyncClient() as client:
            await ConfigController(settings, Registry(()), client).command(settings.replicas[0], "start", settings)
        command, options = calls[0]
        assert "vllm.entrypoints.openai.api_server" in command
        assert command[-2:] == ["--quantization", "moqe_ascend_int4"]
        assert options["env"]["ASCEND_RT_VISIBLE_DEVICES"] == "3"
        assert options["env"]["FLAG"] == "enabled"
    asyncio.run(scenario())
