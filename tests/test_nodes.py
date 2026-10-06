import asyncio
from dataclasses import replace

import httpx

from moqe_serving.config import Node, Replica, Settings
from moqe_serving.pool.controller import ConfigController
from moqe_serving.pool.registry import Registry
from test_admission import backend


def test_disabled_node_drains_all_replicas_before_waiting():
    async def scenario():
        nodes = (Node('old', 'old'), Node('other', 'other'))
        replicas = tuple(Replica(str(i), 'awq', f'http://good{i}/v1', 'backend',
                                 node_id='old' if i < 2 else 'other', device_ids=(i,)) for i in range(3))
        settings = Settings(replicas=replicas, nodes=nodes)
        registry = Registry(())
        async with httpx.AsyncClient(transport=httpx.MockTransport(backend)) as client:
            controller = ConfigController(settings, registry, client)
            await controller.reconcile(settings)
            lease = registry.acquire('awq')
            assert lease.node_id == 'old'
            desired = replace(settings, nodes=(replace(nodes[0], enabled=False), nodes[1]))
            task = asyncio.create_task(controller.reconcile(desired))
            await asyncio.sleep(0.01)
            assert registry.states[lease.id] == 'draining'
            assert registry.states['1'] == 'offline'
            # All replicas of the removed node stop new leases immediately.
            new_lease = registry.acquire('awq')
            assert new_lease.node_id == 'other'
            registry.release(new_lease)
            registry.release(lease)
            await task
            assert all(registry.states[str(i)] == 'offline' for i in (0, 1))
            await controller.reconcile(settings)
            assert all(registry.states[str(i)] == 'ready' for i in (0, 1, 2))
    asyncio.run(scenario())


def test_add_entire_new_node_and_its_replicas():
    async def scenario():
        node = Node('first', 'first')
        first = Replica('first', 'awq', 'http://good/v1', 'backend', node_id='first', device_ids=(0,))
        settings = Settings(replicas=(first,), nodes=(node,))
        registry = Registry(())
        async with httpx.AsyncClient(transport=httpx.MockTransport(backend)) as client:
            controller = ConfigController(settings, registry, client)
            await controller.reconcile(settings)
            added = Replica('added', 'awq', 'http://second/v1', 'backend', node_id='second', device_ids=(0,))
            await controller.reconcile(replace(settings, nodes=(node, Node('second', 'second')), replicas=(first, added)))
            assert registry.states == {'first': 'ready', 'added': 'ready'}
            assert len(controller.snapshot()['nodes']) == 2
    asyncio.run(scenario())
