import asyncio
from dataclasses import replace
import shlex

import httpx
import pytest

from moqe_serving.config import Launch, Node, Replica, Settings
from moqe_serving.pool.controller import ConfigController
from moqe_serving.pool.registry import Registry
from moqe_serving.vllm_options import vllm_arguments
from test_admission import backend


def test_scalar_boolean_json_and_list_arguments():
    assert vllm_arguments({"max_model_len": 8192, "gpu_memory_utilization": 0.5,
        "enable_prefix_caching": False, "enforce_eager": True, "dtype": None,
        "compilation_config": {"mode": 0}, "served_model_aliases": ["one", "two"]}) == [
        "--max-model-len", "8192", "--gpu-memory-utilization", "0.5",
        "--no-enable-prefix-caching", "--enforce-eager",
        "--compilation-config", '{"mode":0}', "--served-model-aliases", "one", "two"]


@pytest.mark.parametrize("options", [
    {"max_model_len": 0}, {"max_num_seqs": True}, {"gpu_memory_utilization": 1.1},
    {"gpu_memory_utilization": 0}, {"dtype": float("nan")}, {"--dtype": "float16"},
    {"max_model_len": 128, "max-model-len": 256}, {"port": 8000},
])
def test_invalid_options(options):
    with pytest.raises(ValueError):
        vllm_arguments(options)


def test_reject_missing_marker_duplicate_flags_and_unsafe_inline():
    for command in [("start",), ("start", "--max-model-len", "128", "{vllm_args}"),
                    ("start", "--max-model-len=128", "{vllm_args}"),
                    ("start", "prefix {vllm_args}")]:
        with pytest.raises(ValueError):
            Launch(command, ("stop",), vllm_args={"max_model_len": 256})


@pytest.mark.parametrize("shell", [False, True])
def test_real_command_expansion_preserves_values_and_remote_quoting(tmp_path, monkeypatch, shell):
    async def scenario():
        # A shell-sensitive value must arrive at vLLM as one literal argument.
        value = "path with spaces; $(echo forbidden)"
        command = ("bash", "-lc", "exec backend --model {model_path} {vllm_args}") if shell else (
            "backend", "--model", "{model_path}", "{vllm_args}")
        replica = Replica("one", "awq", "http://good/v1", "backend", node_id="node",
            model_path="/models/awq", device_ids=(0,), launch=Launch(command, ("stop",),
            vllm_args={"tokenizer": value, "compilation_config": {"mode": 0}}))
        settings = Settings(replicas=(replica,), nodes=(Node("node", "remote", "root@remote", "container"),),
            lifecycle_log_dir=str(tmp_path))
        captured = []
        class Process:
            returncode = 0
            async def wait(self):
                return 0
        async def spawn(*args, **kwargs):
            captured.append(args)
            return Process()
        monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
        async with httpx.AsyncClient() as client:
            controller = ConfigController(settings, Registry(()), client)
            await controller.command(replica, "start", settings)
            remote = shlex.split(captured[0][-1])
            actual = shlex.split(remote[-1])[1:] if shell else remote[4:]
            assert actual == ["backend", "--model", "/models/awq", "--tokenizer", value,
                              "--compilation-config", '{"mode":0}']
            await controller.command(replica, "stop", settings)
            assert shlex.split(captured[1][-1])[-1] == "stop"
    asyncio.run(scenario())


def test_hyperparameter_change_drains_restarts_and_readmits():
    async def scenario():
        replica = Replica("one", "awq", "http://good/v1", "backend", model_path="/models/awq",
            device_ids=(0,), launch=Launch(("start", "{vllm_args}"), ("stop",),
                                         vllm_args={"max_num_seqs": 8}))
        settings = Settings(replicas=(replica,))
        registry, events = Registry(()), []
        async with httpx.AsyncClient(transport=httpx.MockTransport(backend)) as client:
            controller = ConfigController(settings, registry, client)
            async def command(current, action, config):
                events.append((action, current.launch.vllm_args["max_num_seqs"]))
            controller.command = command
            await controller.reconcile(settings)
            lease = registry.acquire("awq")
            desired = replace(settings, replicas=(replace(replica,
                launch=replace(replica.launch, vllm_args={"max_num_seqs": 16})),))
            pending = asyncio.create_task(controller.reconcile(desired))
            await asyncio.sleep(0.01)
            assert registry.states["one"] == "draining"
            assert events == [("start", 8)]
            registry.release(lease)
            await pending
            assert events == [("start", 8), ("stop", 8), ("start", 16)]
            assert registry.states["one"] == "ready"
    asyncio.run(scenario())
