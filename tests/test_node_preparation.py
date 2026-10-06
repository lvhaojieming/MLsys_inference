"""Preparation blocks admission on failure and runs once per pending node."""
import asyncio
from dataclasses import replace
import json

import httpx
import pytest

from moqe_serving.config import Launch, Node, Replica, Settings
from moqe_serving.deployment.config import NodePreparation
from moqe_serving.deployment.prepare import DEPLOY_HELPER, NodePreparer
from moqe_serving.pool.controller import ConfigController
from moqe_serving.pool.registry import Registry
from moqe_serving.pool.health import HealthManager
from admission_support import backend


def managed(replica_id, node_id, device):
    return Replica(replica_id, "awq", f"http://{replica_id}/v1", "backend", node_id=node_id,
        device_ids=(device,), model_path="/models/awq", launch=Launch(
            ("python3", "{backend_helper}", "start"), ("python3", "{backend_helper}", "stop")))


def test_preparation_sequence_and_helper_payload(tmp_path, monkeypatch):
    async def scenario():
        calls = []
        async def run(node, command, **kwargs):
            calls.append((command, kwargs))
            return ""
        monkeypatch.setattr("moqe_serving.deployment.prepare.run_node_command", run)
        node = Node("node", "remote", prepare=NodePreparation(host_checks=(("npu-smi", "info"),),
            runtime_checks=(("python3", "-c", "assert True"),), required_paths=("/adapter",)))
        report = await NodePreparer(tmp_path).prepare(node, (managed("one", "node", 0), managed("two", "node", 1)))
        assert report["passed"]
        assert [r["name"] for r in report["checks"]] == [
            "communication", "host-0", "runtime", "runtime-0", "helper", "helper-check"]
        spec = json.loads(calls[2][0][-1])
        assert spec["paths"] == ["/adapter", "/models/awq"]  # Deduplicated across replicas.
        assert spec["modules"] == ["vllm"]
        assert not calls[0][1]["in_container"]
        assert calls[2][1]["in_container"]
        import base64
        helper = base64.b64decode(calls[4][0][-1])
        compile(helper, "deployed_helper", "exec")
        assert helper.startswith(b"# Managed by MoQE\n")
    asyncio.run(scenario())


def test_first_deploy_hot_add_failure_and_retry(tmp_path, monkeypatch):
    async def scenario():
        events, failures = [], set()
        async def prepare(self, node, replicas):
            events.append(("prepare", node.id))
            return {"passed": node.id not in failures, "failed_check": "runtime", "error": "missing module"}
        monkeypatch.setattr(NodePreparer, "prepare", prepare)
        old = Node("old", "old", prepare=NodePreparation())
        first = managed("first", "old", 0)
        settings = Settings(nodes=(old,), replicas=(first,), lifecycle_log_dir=str(tmp_path))
        registry = Registry(())
        async with httpx.AsyncClient(transport=httpx.MockTransport(backend)) as client:
            controller = ConfigController(settings, registry, client)
            async def command(replica, action, config):
                events.append((action, replica.id))
            controller.command = command
            await controller.reconcile(settings)
            await controller.reconcile(settings)
            assert events == [("prepare", "old"), ("start", "first")]
            new = Node("new", "new", prepare=NodePreparation())
            desired = replace(settings, nodes=(old, new), replicas=(first, managed("second", "new", 0),
                                                                      managed("third", "new", 1)))
            failures.add("new")
            await controller.reconcile(desired)
            assert registry.states == {"first": "ready", "second": "unhealthy", "third": "unhealthy"}
            assert events[-1] == ("prepare", "new")  # No backend launches after a failed preflight.
            health = HealthManager(desired, registry, client)
            for _ in range(3):
                await health.check(registry.get("second"))
            assert registry.states["second"] == "unhealthy"  # Even a healthy HTTP endpoint cannot bypass preparation.
            assert controller.snapshot()["nodes"][1]["preparation"]["passed"] is False
            disabled = replace(desired, nodes=(old, replace(new, enabled=False)))
            await controller.reconcile(disabled)
            failures.clear()
            await controller.reconcile(desired)
            assert all(state == "ready" for state in registry.states.values())
            assert events[-3:] == [("prepare", "new"), ("start", "second"), ("start", "third")]
    asyncio.run(scenario())


def test_runtime_failure_has_stage_and_does_not_deploy_helper(tmp_path, monkeypatch):
    async def scenario():
        calls = []
        async def run(node, command, **kwargs):
            calls.append(command)
            if len(calls) == 2:
                raise RuntimeError("Required path missing: /models/awq")
        monkeypatch.setattr("moqe_serving.deployment.prepare.run_node_command", run)
        node = Node("node", "remote", prepare=NodePreparation())
        report = await NodePreparer(tmp_path).prepare(node, (managed("one", "node", 0),))
        assert not report["passed"]
        assert report["failed_check"] == "runtime"
        assert len(calls) == 2
    asyncio.run(scenario())


def test_atomic_helper_install_is_idempotent_and_preserves_unmanaged_files(tmp_path):
    import base64
    import subprocess
    import sys
    path = tmp_path / "helper.py"
    source = b"# Managed by MoQE\nprint('hello')\n"
    command = [sys.executable, "-c", DEPLOY_HELPER, str(path), base64.b64encode(source).decode()]
    subprocess.run(command, check=True, capture_output=True)
    before = path.stat().st_mtime_ns
    subprocess.run(command, check=True, capture_output=True)
    assert path.stat().st_mtime_ns == before
    path.write_text("unrelated user file")
    result = subprocess.run(command, capture_output=True)
    assert result.returncode != 0
    assert path.read_text() == "unrelated user file"


def test_prepare_config_loading_and_placeholder_validation(tmp_path):
    example = Settings.load('configs/first_deployment.example.json')
    assert len(example.replicas) == 2 and all(r.launch for r in example.replicas)
    assert all(node.prepare for node in example.nodes)
    value = {"nodes": [{"id": "node", "host": "remote", "prepare": {
        "host_checks": [["npu-smi", "info"]], "required_modules": ["vllm", "torch_npu"]}}]}
    path = tmp_path / "cluster.json"
    path.write_text(json.dumps(value))
    settings = Settings.load(path)
    assert settings.nodes[0].prepare.host_checks == (("npu-smi", "info"),)
    with pytest.raises(ValueError, match="backend_helper"):
        Settings(nodes=(Node("node", "remote"),), replicas=(managed("one", "node", 0),))
    value["nodes"][0]["prepare"]["host_checks"] = ["not an argv array"]
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        Settings.load(path)
