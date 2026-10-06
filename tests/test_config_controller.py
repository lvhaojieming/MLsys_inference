import asyncio
from dataclasses import replace
import json
import time

import httpx
from fastapi.testclient import TestClient
import pytest

from moqe_serving.config import Admission, Launch, Node, Replica, Settings
from moqe_serving.gateway.app import create_app
from moqe_serving.pool.controller import ConfigController
from moqe_serving.pool.registry import Registry
from admission_support import backend


def test_file_edit_add_disable_and_invalid_edit(tmp_path, monkeypatch):
    monkeypatch.setenv('ADMIN', 'secret')
    path = tmp_path / 'config.json'
    one = {'id': 'one', 'expert': 'awq', 'base_url': 'http://good/v1', 'model': 'backend'}
    config = {'replicas': [one], 'watch_config': True, 'admission_enabled': True,
              'admin_token_env': 'ADMIN', 'config_poll_interval_seconds': 60}
    path.write_text(json.dumps(config))
    auth = {'Authorization': 'Bearer secret'}
    app = create_app(Settings.load(path), httpx.MockTransport(backend), config_path=str(path))
    with TestClient(app) as client:
        two = {**one, 'id': 'two', 'base_url': 'http://second/v1'}
        config['replicas'].append(two)
        path.write_text(json.dumps(config))
        assert client.post('/admin/config/reload', headers=auth).status_code == 200
        assert app.state.registry.states == {'one': 'ready', 'two': 'ready'}
        # Adding an unknown expert is rejected before removing the existing replicas.
        config['replicas'] = [{**two, 'expert': 'int8'}]
        path.write_text(json.dumps(config))
        assert client.post('/admin/config/reload', headers=auth).status_code == 400
        assert app.state.registry.states == {'one': 'ready', 'two': 'ready'}
        config['replicas'] = [{**one, 'enabled': False}, two]
        path.write_text(json.dumps(config))
        assert client.post('/admin/config/reload', headers=auth).status_code == 200
        assert app.state.registry.states['one'] == 'offline'
        result = client.post('/v1/chat/completions', json={'model': 'awq', 'messages': [{}]})
        assert result.headers['x-moqe-replica'] == 'two'
        # Re-enabling a drained replica repeats admission.
        config['replicas'] = [one, two]
        path.write_text(json.dumps(config))
        assert client.post('/admin/config/reload', headers=auth).status_code == 200
        assert app.state.registry.states['one'] == 'ready'


def test_managed_launch_admit_drain_stop_order():
    async def scenario():
        replica = Replica('one', 'awq', 'http://good/v1', 'backend', node_id='node',
                          device_ids=(3,), model_path='/models/awq',
                          launch=Launch(('start',), ('stop',)))
        settings = Settings(replicas=(replica,), nodes=(Node('node', 'localhost'),), admission_enabled=True)
        registry = Registry(())
        events = []
        async with httpx.AsyncClient(transport=httpx.MockTransport(backend)) as client:
            controller = ConfigController(settings, registry, client)
            async def command(r, action, configuration):
                events.append((action, registry.states[r.id], registry.inflight[r.id]))
            controller.command = command
            await controller.reconcile(settings)
            assert events == [('start', 'starting', 0)]
            assert registry.states['one'] == 'ready'
            lease = registry.acquire('awq')
            removal = asyncio.create_task(controller.reconcile(replace(settings, replicas=())))
            await asyncio.sleep(0.01)
            assert registry.states['one'] == 'draining'
            assert len(events) == 1  # Must not stop backend with an active request.
            registry.release(lease)
            await removal
            assert events[-1] == ('stop', 'offline', 0)
            assert not controller.owned
    asyncio.run(scenario())


def test_custom_admission_parameters():
    def custom(request):
        if request.method == 'POST':
            value = json.loads(request.content)
            assert value['messages'][0]['content'] == 'custom question'
            assert value['max_tokens'] == 64
        return backend(request)
    settings = Settings(replicas=(Replica('one', 'awq', 'http://good/v1', 'backend'),),
                        admission_enabled=True,
                        admission=Admission(prompt='custom question', max_tokens=64))
    app = create_app(settings, httpx.MockTransport(custom))
    with TestClient(app):
        assert app.state.registry.states['one'] == 'ready'


def test_example_loads_and_config_validates():
    settings = Settings.load('configs/pool_lifecycle.example.json')
    assert settings.gateway.port == 18081
    assert settings.replicas[1].launch.start_command
    cluster = Settings.load('configs/awq_pool_cluster.json')
    assert cluster.replicas[0].model_path.endswith('Qwen3-14B-AWQ-Ascend-INT4')
    with pytest.raises(ValueError):
        replace(settings, nodes=())


def test_watcher_applies_edit_without_management_call(tmp_path):
    path = tmp_path / 'config.json'
    one = {'id': 'one', 'expert': 'awq', 'base_url': 'http://good/v1', 'model': 'backend'}
    config = {'replicas': [one], 'watch_config': True, 'admission_enabled': True,
              'config_poll_interval_seconds': 0.01}
    path.write_text(json.dumps(config))
    app = create_app(Settings.load(path), httpx.MockTransport(backend), config_path=str(path))
    with TestClient(app):
        config['replicas'].append({**one, 'id': 'two', 'base_url': 'http://second/v1'})
        temp = path.with_suffix('.tmp')
        temp.write_text(json.dumps(config))
        # Windows editors must retry atomic replacement while a watcher has the
        # old file briefly open; Linux permits the rename immediately.
        replace_deadline = time.monotonic() + 2
        while True:
            try:
                temp.replace(path)
                break
            except PermissionError:
                if time.monotonic() >= replace_deadline:
                    raise
                time.sleep(0.01)
        deadline = time.monotonic() + 2
        while app.state.registry.states.get('two') != 'ready' and time.monotonic() < deadline:
            time.sleep(0.01)
        assert app.state.registry.states.get('two') == 'ready'
