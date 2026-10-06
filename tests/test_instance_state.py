import asyncio

import httpx
import pytest

from moqe_serving.config import Replica, Settings
from moqe_serving.pool.admission import validate_replica
from moqe_serving.pool.models import InstanceState
from moqe_serving.pool.registry import Registry
from admission_support import backend


def replica():
    return Replica('one', 'awq', 'http://good/v1', 'backend')


def test_exact_document_states_and_no_admission_bypass():
    assert {s.value for s in InstanceState} == {'starting', 'ready', 'draining', 'unhealthy', 'offline'}
    with pytest.raises(ValueError):
        Settings(replicas=(replica(),), admission_enabled=False)
    with pytest.raises(ValueError):
        Registry((replica(),), require_admission=False)


def test_normal_drain_always_records_both_transitions():
    registry = Registry((replica(),))
    registry.transition('one', 'ready', 'test admission passed')
    registry.drain('one')
    assert [(e['from'], e['to']) for e in registry.transitions] == [
        ('starting', 'ready'), ('ready', 'draining'), ('draining', 'offline')]
    with pytest.raises(ValueError):
        registry.transition('one', 'ready', 'attempt to bypass restart and validation')


def test_unhealthy_is_not_scheduled_and_recovers_after_validation():
    async def scenario():
        registry = Registry((replica(),))
        async with httpx.AsyncClient(transport=httpx.MockTransport(backend)) as client:
            await validate_replica(registry, replica(), client, 1)
            registry.transition('one', 'unhealthy', 'health check failed')
            assert registry.ready_instances('awq') == []
            with pytest.raises(KeyError):
                registry.acquire('awq')
            await validate_replica(registry, replica(), client, 1)
        assert registry.states['one'] == InstanceState.READY
        assert registry.transitions[-1]['from'] == 'unhealthy'
        assert registry.transitions[-1]['to'] == 'ready'
    asyncio.run(scenario())


def test_drain_wins_over_concurrent_successful_admission():
    async def scenario():
        reached = asyncio.Event()
        proceed = asyncio.Event()
        async def delayed(request):
            if request.method == 'GET':
                reached.set()
                await proceed.wait()
            return backend(request)
        registry = Registry((replica(),))
        async with httpx.AsyncClient(transport=httpx.MockTransport(delayed)) as client:
            validation = asyncio.create_task(validate_replica(registry, replica(), client, 2))
            await reached.wait()
            assert registry.states['one'] == InstanceState.STARTING
            assert 'one' in registry.validating
            with pytest.raises(ValueError):
                registry.begin_validation('one')
            registry.drain('one')
            proceed.set()
            await validation
        assert registry.states['one'] == InstanceState.OFFLINE
        assert not registry.validating
        assert not registry.ready_instances('awq')
    asyncio.run(scenario())
