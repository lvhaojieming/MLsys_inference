import asyncio

import httpx

from moqe_serving.config import Health, Replica, Settings
from moqe_serving.pool.admission import validate_replica
from moqe_serving.pool.health import HealthManager
from moqe_serving.pool.registry import Registry
from admission_support import backend


def test_failure_threshold_and_correctness_gated_recovery():
    async def scenario():
        status = {'up': True, 'correct': True}
        replica = Replica('one', 'awq', 'http://good/v1', 'backend')
        def transport(request):
            if not status['up']:
                return httpx.Response(503)
            if not status['correct'] and request.method == 'POST':
                return httpx.Response(200, json={'choices': [{'message': {'content': '0'}}]})
            return backend(request)
        settings = Settings(replicas=(replica,), health=Health(failure_threshold=2, recovery_threshold=2))
        registry = Registry((replica,))
        async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
            await validate_replica(registry, replica, client, 1)
            manager = HealthManager(settings, registry, client)
            status['up'] = False
            await manager.check(replica)
            assert registry.states['one'] == 'ready'
            await manager.check(replica)
            assert registry.states['one'] == 'unhealthy'
            assert registry.ready_instances('awq') == []
            status.update(up=True, correct=False)
            await manager.check(replica)
            await manager.check(replica)
            assert registry.states['one'] == 'unhealthy'  # Liveness is not correctness.
            status['correct'] = True
            await manager.check(replica)
            assert registry.states['one'] == 'unhealthy'
            await manager.check(replica)
            assert registry.states['one'] == 'ready'
            registry.drain('one')
            await manager.check(replica)
            assert registry.states['one'] == 'offline'
    asyncio.run(scenario())


def test_drain_during_health_probe_does_not_revive_instance():
    async def scenario():
        replica = Replica('one', 'awq', 'http://good/v1', 'backend')
        registry = Registry((replica,))
        registry.transition('one', 'ready', 'test admission passed')
        registry.transition('one', 'unhealthy', 'test failure')
        reached, proceed = asyncio.Event(), asyncio.Event()
        async def delayed(request):
            reached.set()
            await proceed.wait()
            return backend(request)
        settings = Settings(replicas=(replica,), health=Health(recovery_threshold=1))
        async with httpx.AsyncClient(transport=httpx.MockTransport(delayed)) as client:
            manager = HealthManager(settings, registry, client)
            check = asyncio.create_task(manager.check(replica))
            await reached.wait()
            registry.drain('one')
            proceed.set()
            await check
            assert registry.states['one'] == 'offline'
    asyncio.run(scenario())
