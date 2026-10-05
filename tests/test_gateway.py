import httpx
import pytest
from fastapi.testclient import TestClient

from moqe_serving.config import Replica, Settings
from moqe_serving.gateway.app import create_app
from moqe_serving.pool.registry import Registry


def test_health_does_not_claim_backend_readiness():
    with TestClient(create_app(Settings())) as client:
        assert client.get("/health").json() == {"status": "ok", "configured_replicas": 0}
        assert client.get("/v1/models").json()["data"] == []
        assert client.post("/v1/chat/completions", json={"model": "auto", "messages": []}).status_code == 400


def test_replica_leases_and_balancing():
    replicas = tuple(Replica(str(i), "awq", "http://localhost/v1", "backend") for i in range(2))
    registry = Registry(replicas)
    first, second = registry.acquire("awq"), registry.acquire("awq")
    assert first != second
    registry.release(first)
    assert registry.acquire("awq") == first
    with pytest.raises(KeyError):
        registry.acquire("unknown")


@pytest.mark.parametrize("stream", [False, True])
def test_proxy_and_release(stream):
    def backend(request):
        import json
        assert json.loads(request.content)["model"] == "backend-model"
        if stream:
            return httpx.Response(200, content=b'data: {"choices": []}\n\ndata: [DONE]\n\n')
        return httpx.Response(200, json={"model": "backend-model", "choices": []})
    settings = Settings((Replica("r1", "awq", "http://backend/v1", "backend-model"),))
    app = create_app(settings, httpx.MockTransport(backend))
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json={
            "model": "awq", "messages": [{"role": "user", "content": "hello"}], "stream": stream,
        })
        assert response.status_code == 200
        assert response.headers["x-moqe-replica"] == "r1"
        assert app.state.registry.inflight["r1"] == 0
        if stream:
            assert "[DONE]" in response.text
        else:
            assert response.json()["model"] == "awq"


@pytest.mark.parametrize("failure", ["status", "connection", "invalid_json"])
def test_backend_errors_release_lease(failure):
    def backend(request):
        if failure == "connection":
            raise httpx.ConnectError("offline", request=request)
        if failure == "status":
            return httpx.Response(503)
        return httpx.Response(200, content=b"not json")
    settings = Settings((Replica("r1", "awq", "http://backend/v1", "backend-model"),))
    app = create_app(settings, httpx.MockTransport(backend))
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json={
            "model": "awq", "messages": [{"role": "user", "content": "hello"}],
        })
        assert response.status_code == 502
        assert app.state.registry.inflight["r1"] == 0


def test_duplicate_ids_rejected():
    replica = Replica("same", "awq", "http://backend/v1", "model")
    with pytest.raises(ValueError):
        Settings((replica, replica))
