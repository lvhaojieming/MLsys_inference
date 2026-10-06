"""Reconcile a configuration file with admitted replicas in one gateway process."""
import asyncio
from dataclasses import replace
import logging
import os
from pathlib import Path
import shlex
import time

from ..config import Settings
from ..vllm_options import vllm_arguments
from .admission import validate_replica

logger = logging.getLogger(__name__)


class ConfigController:
    def __init__(self, settings, registry, client, path=None):
        self.settings = settings
        self.baseline = settings
        self.registry = registry
        self.client = client
        self.path = path
        self.lock = asyncio.Lock()
        self.applied = {}
        self.owned = set()
        self.last_error = None
        self.last_signature = None
        self.expert_models = {}
        for replica in settings.replicas:
            self.expert_models.setdefault(replica.expert, set()).add(replica.model)

    async def command(self, replica, action, settings):
        node = next((n for n in settings.nodes if n.id == replica.node_id), None)
        values = {"id": replica.id, "expert": replica.expert, "model": replica.model,
                  "model_path": replica.model_path or "", "port": str(replica.backend_port or ""),
                  "device_ids": ",".join(map(str, replica.device_ids)),
                  "host": node.host if node else "", "container": (node.container or "") if node else ""}
        def expand(value):
            for key, replacement in values.items():
                value = value.replace("{" + key + "}", replacement)
            return value
        arguments = vllm_arguments(replica.launch.vllm_args) if action == "start" else []
        command = []
        for part in getattr(replica.launch, action + "_command"):
            if part == "{vllm_args}":
                command.extend(arguments)
            else:
                # Insert quoted argv only into an explicitly configured shell.
                # Do this after ordinary expansion so JSON braces remain literal.
                command.append(expand(part).replace("{vllm_args}", shlex.join(arguments)))
        env = {key: expand(value) for key, value in replica.launch.env.items()}
        if node and node.container:
            command = ["docker", "exec", node.container, "env", *[f"{k}={v}" for k, v in env.items()], *command]
        elif node and node.ssh_target:
            command = ["env", *[f"{k}={v}" for k, v in env.items()], *command]
        if node and node.ssh_target:
            command = ["ssh", *node.ssh_options, node.ssh_target, shlex.join(command)]
        log_dir = Path(settings.lifecycle_log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        # Avoid interpreting replica ids as filesystem paths.
        name = replica.id.encode().hex()
        with (log_dir / f"{name}-{action}.log").open("ab") as log:
            process = await asyncio.create_subprocess_exec(*command, stdout=log, stderr=log,
                                                           env={**os.environ, **env})
            try:
                await asyncio.wait_for(process.wait(), replica.launch.command_timeout_seconds)
            except BaseException:
                if process.returncode is None:
                    process.kill()
                    await process.wait()
                raise
            if process.returncode:
                raise RuntimeError(f"Replica {replica.id}: {action} exited {process.returncode}")

    async def remove(self, replica, settings):
        self.registry.drain(replica.id)
        start = time.monotonic()
        while self.registry.inflight[replica.id]:
            if time.monotonic() - start > settings.drain_timeout_seconds:
                raise TimeoutError(f"Drain timed out for {replica.id}; backend kept running")
            await asyncio.sleep(min(0.2, settings.config_poll_interval_seconds))
        if replica.id in self.owned:
            await self.command(replica, "stop", self.settings)
            self.owned.remove(replica.id)
        self.applied.pop(replica.id, None)

    async def reconcile(self, desired):
        async with self.lock:
            # Validate the entire edit before draining or launching anything.
            if replace(desired, replicas=self.baseline.replicas, nodes=self.baseline.nodes) != self.baseline:
                raise ValueError("Only nodes and replicas can be hot-edited; other settings require restart")
            for r in desired.replicas:
                if r.expert not in self.expert_models or r.model not in self.expert_models[r.expert]:
                    raise ValueError("Hot edits may only add replicas of existing experts and model names")
            active = {r.id: r for r in desired.active_replicas}
            removing = []
            for replica_id, current in list(self.applied.items()):
                target = active.get(replica_id)
                old_node = next((n for n in self.settings.nodes if n.id == current.node_id), None)
                new_node = next((n for n in desired.nodes if n.id == current.node_id), None)
                if target != current or old_node != new_node:
                    removing.append(current)
            # Withdraw every replica on a disabled node immediately, before
            # waiting for any one replica's long-running request to finish.
            for current in removing:
                self.registry.drain(current.id)
            for current in removing:
                await self.remove(current, desired)
            # Use new node definitions only after removing old managed services.
            self.settings = desired
            for replica_id, replica in active.items():
                if replica_id in self.applied:
                    continue
                if replica_id in self.registry.states:
                    self.registry.replace_offline(replica)
                else:
                    self.registry.add(replica)
                try:
                    if replica.launch:
                        # If launch partially succeeds, retain ownership for a later stop.
                        self.owned.add(replica_id)
                        await self.command(replica, "start", desired)
                    result = await validate_replica(self.registry, replica, self.client,
                                                    desired.admission_timeout_seconds, desired.admission)
                    self.applied[replica_id] = replica
                    if not result["passed"]:
                        logger.error("Replica %s admission failed: %s", replica_id, result)
                except Exception as exc:
                    self.registry.transition(replica_id, "unhealthy", "configured launch failed")
                    self.registry.validation[replica_id] = {"passed": False, "error": str(exc)}
                    self.applied[replica_id] = replica
                    logger.exception("Replica %s launch failed", replica_id)
            self.last_error = None

    async def reload(self):
        if not self.path:
            raise ValueError("No source configuration path")
        desired = Settings.load(self.path)
        await self.reconcile(desired)

    async def watch(self):
        while True:
            await asyncio.sleep(self.settings.config_poll_interval_seconds)
            try:
                contents = Path(self.path).read_bytes()
                if contents == self.last_signature:
                    continue
                self.last_signature = contents
                await self.reload()
                logger.info("Config replicas reconciled: %s", self.path)
            except Exception as exc:
                self.last_error = str(exc)
                logger.error("Config edit rejected: %s", exc)

    def snapshot(self):
        instances = self.registry.snapshot()
        nodes = [{"id": n.id, "host": n.host, "enabled": n.enabled,
                  "replicas": [r.id for r in self.settings.replicas if r.node_id == n.id],
                  "ready_replicas": [r["id"] for r in instances
                                     if self.registry.get(r["id"]).node_id == n.id and r["state"] == "ready"],
                  "inflight": sum(r["inflight"] for r in instances
                                  if self.registry.get(r["id"]).node_id == n.id)}
                 for n in self.settings.nodes]
        return {"config_path": self.path, "last_error": self.last_error,
                "managed_replicas": sorted(self.owned), "nodes": nodes, "instances": instances,
                "transitions": self.registry.transitions}
