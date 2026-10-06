import httpx
from fastapi.testclient import TestClient

from moqe_serving.config import Replica, Settings
from moqe_serving.gateway.app import create_app
from moqe_serving.pool.registry import Registry
from admission_support import backend


def test_startup_admits_only_correct_backend():
    settings = Settings(replicas=(Replica('good', 'awq', 'http://good/v1', 'backend'),
                                 Replica('bad', 'gptq', 'http://wrong/v1', 'backend')),
                        admission_enabled=True)
    app = create_app(settings, httpx.MockTransport(backend))
    with TestClient(app) as client:
        assert app.state.registry.states == {'good': 'ready', 'bad': 'unhealthy'}
        assert client.get('/ready').status_code == 503
        assert client.post('/v1/chat/completions', json={"model": "gptq", "messages": [{}]}).status_code == 503


def test_hot_add_and_drain_without_restart(monkeypatch):
    monkeypatch.setenv('TEST_ADMIN_TOKEN', 'test-secret')
    initial = Replica('existing', 'awq', 'http://good/v1', 'backend')
    app = create_app(Settings(replicas=(initial,), admission_enabled=True,
                              admin_token_env='TEST_ADMIN_TOKEN'), httpx.MockTransport(backend))
    auth = {'Authorization': 'Bearer test-secret'}
    replica = {'id': 'added', 'expert': 'awq', 'base_url': 'http://good/v1', 'model': 'backend'}
    with TestClient(app) as client:
        assert client.post('/admin/instances', json=replica).status_code == 401
        assert client.post('/admin/instances', json={**replica, 'expert': 'int8'}, headers=auth).status_code == 400
        assert client.post('/admin/instances', json={**replica, 'model': 'different'}, headers=auth).status_code == 400
        added = client.post('/admin/instances', json=replica, headers=auth)
        assert added.status_code == 201
        assert added.json()['state'] == 'ready'
        assert client.post('/admin/instances', json=replica, headers=auth).status_code == 409
        assert client.get('/ready').status_code == 200
        assert client.post('/admin/instances/existing/drain', headers=auth).json()['state'] == 'offline'
        response = client.post('/v1/chat/completions', json={"model": "awq", "messages": [{}]})
        assert response.status_code == 200
        assert response.headers['x-moqe-replica'] == 'added'
        assert client.post('/admin/instances/added/drain', headers=auth).json()['state'] == 'offline'
        assert client.post('/v1/chat/completions', json={"model": "awq", "messages": [{}]}).status_code == 503
        assert client.post('/admin/instances/added/validate', headers=auth).json()['state'] == 'ready'


def test_failed_hot_add_keeps_existing_pool_serving(monkeypatch):
    monkeypatch.setenv('TEST_ADMIN_TOKEN', 'test-secret')
    initial = Replica('existing', 'awq', 'http://good/v1', 'backend')
    app = create_app(Settings(replicas=(initial,), admission_enabled=True,
                              admin_token_env='TEST_ADMIN_TOKEN'), httpx.MockTransport(backend))
    with TestClient(app) as client:
        failed = client.post('/admin/instances', headers={'Authorization': 'Bearer test-secret'},
                             json={'id': 'bad', 'expert': 'awq', 'base_url': 'http://wrong/v1', 'model': 'backend'})
        assert failed.status_code == 422
        assert app.state.registry.states['bad'] == 'unhealthy'
        for _ in range(3):
            response = client.post('/v1/chat/completions', json={"model": "awq", "messages": [{}]})
            assert response.status_code == 200
            assert response.headers['x-moqe-replica'] == 'existing'
        assert client.get('/ready').status_code == 200


def test_validation_and_draining_never_take_new_leases():
    replica = Replica('r', 'awq', 'http://good/v1', 'backend')
    registry = Registry((replica,), require_admission=True)
    import pytest
    with pytest.raises(KeyError):
        registry.acquire('awq')
    registry.transition('r', 'ready', 'test admission passed')
    lease = registry.acquire('awq')
    registry.drain('r')
    assert registry.states['r'] == 'draining'
    with pytest.raises(KeyError):
        registry.acquire('awq')
    registry.release(lease)
    assert registry.states['r'] == 'offline'


def test_truncated_stream_fails_admission():
    def truncated(request):
        response = backend(request)
        if request.method == 'POST' and json.loads(request.content).get('stream'):
            return httpx.Response(200, text='data: {"choices":[{"delta":{"content":"42"}}]}\n\n')
        return response
    app = create_app(Settings(replicas=(Replica('r', 'awq', 'http://good/v1', 'backend'),),
                              admission_enabled=True), httpx.MockTransport(truncated))
    with TestClient(app):
        assert app.state.registry.states['r'] == 'unhealthy'
        assert not app.state.registry.validation['r']['passed']
