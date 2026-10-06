"""Periodic failure/recovery thresholds; recovery still requires real admission."""
import asyncio
import os

import anyio

from .admission import validate_replica
from .models import InstanceState


class HealthManager:
    def __init__(self, settings, registry, client):
        self.settings = settings
        self.registry = registry
        self.client = client
        self.counts = {}

    async def probe(self, replica):
        headers = {}
        if replica.api_key_env:
            key = os.environ.get(replica.api_key_env)
            if not key:
                return False
            headers['Authorization'] = 'Bearer ' + key
        try:
            with anyio.fail_after(self.settings.health.probe_timeout_seconds):
                response = await self.client.get(replica.base_url.rstrip('/') + '/models', headers=headers)
                response.raise_for_status()
                return replica.model in {m.get('id') for m in response.json().get('data', [])}
        except Exception:
            return False

    async def check(self, replica):
        state = self.registry.states.get(replica.id)
        if state not in {InstanceState.READY, InstanceState.UNHEALTHY} or replica.id in self.registry.validating:
            self.counts.pop(replica.id, None)
            return
        healthy = await self.probe(replica)
        # Config editing/drain may happen while the probe is awaiting the network.
        if self.registry.get(replica.id) != replica or self.registry.states[replica.id] != state:
            self.counts.pop(replica.id, None)
            return
        previous = self.counts.get(replica.id)
        count = previous[2] + 1 if previous and previous[:2] == (replica, state) else 1
        if state == InstanceState.READY:
            if healthy:
                self.counts.pop(replica.id, None)
                return
            self.counts[replica.id] = (replica, state, count)
            if count >= self.settings.health.failure_threshold:
                self.registry.transition(replica.id, InstanceState.UNHEALTHY, 'health failure threshold reached')
                self.counts.pop(replica.id, None)
        else:
            if not healthy:
                self.counts.pop(replica.id, None)
                return
            self.counts[replica.id] = (replica, state, count)
            if count >= self.settings.health.recovery_threshold:
                self.counts.pop(replica.id, None)
                # /models alone is not sufficient evidence of correct generation.
                try:
                    await validate_replica(self.registry, replica, self.client,
                                           self.settings.admission_timeout_seconds, self.settings.admission)
                except ValueError:
                    pass  # A concurrent validation/drain takes precedence.

    async def run(self):
        tasks = {}
        try:
            while True:
                await asyncio.sleep(self.settings.health.interval_seconds)
                for replica in self.registry.replicas:
                    previous = tasks.get(replica.id)
                    if previous and not previous.done():
                        continue
                    if previous:
                        previous.result()
                    tasks[replica.id] = asyncio.create_task(self.check(replica))
        finally:
            for task in tasks.values():
                task.cancel()
            await asyncio.gather(*tasks.values(), return_exceptions=True)
