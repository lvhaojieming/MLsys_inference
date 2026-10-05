import json
import httpx
import pytest
from fastapi.testclient import TestClient
from moqe_serving.config import Replica, Settings
from moqe_serving.gateway.app import create_app
from moqe_serving.routing.runtime import RoutingDecision


class FakeRouter:
    chat_template = "canonical-template"

    def __init__(self):
        self.calls = 0

    def route(self, messages, generation, kwargs):
        self.calls += 1
        assert generation == 128
        assert kwargs == {"enable_thinking": False}
        if messages[0]["content"] == "too long":
            raise ValueError("Prompt exceeds limit; no truncation")
        return RoutingDecision("gptq", {"awq": 0.2, "gptq": 0.8}, 28, 7.0)


@pytest.mark.parametrize("stream", [False, True])
def test_auto_selects_expert_once_and_forwards_canonical_template(stream):
    def backend(request):
        payload = json.loads(request.content)
        assert request.url.host == "gptq"
        assert payload["model"] == "gptq-model"
        assert payload["chat_template"] == "canonical-template"
        assert payload["chat_template_kwargs"] == {"enable_thinking": False}
        if stream:
            return httpx.Response(200, content=b'data: {"model":"gptq-model","choices":[]}\n\ndata: [DONE]\n\n')
        return httpx.Response(200, json={"model": "gptq-model", "choices": []})
    settings = Settings(tuple(Replica(k, k, f"http://{k}/v1", f"{k}-model") for k in ["awq", "gptq"]))
    runtime = FakeRouter()
    app = create_app(settings, httpx.MockTransport(backend), runtime)
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json={"model": "auto", "stream": stream,
                                "messages": [{"role": "user", "content": "hello"}]})
        assert response.status_code == 200
        assert response.headers["x-moqe-expert"] == "gptq"
        assert response.headers["x-moqe-input-tokens"] == "28"
        assert runtime.calls == 1
        assert app.state.registry.inflight == {"awq": 0, "gptq": 0}
        if stream:
            assert '[DONE]' in response.text
            assert '"model": "auto"' in response.text
        else:
            assert response.json()["model"] == "auto"
        assert "auto" in [m["id"] for m in client.get("/v1/models").json()["data"]]


def test_auto_rejects_overlength_before_backend_call():
    def backend(request):
        pytest.fail("Invalid prompt must never reach backend")
    settings = Settings((Replica("r", "gptq", "http://gptq/v1", "m"),))
    app = create_app(settings, httpx.MockTransport(backend), FakeRouter())
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json={"model": "auto",
                                "messages": [{"role": "user", "content": "too long"}]})
        assert response.status_code == 400
        assert app.state.registry.inflight["r"] == 0


def test_auto_without_runtime_fails_explicitly():
    with TestClient(create_app(Settings())) as client:
        response = client.post("/v1/chat/completions", json={"model": "auto",
                                "messages": [{"role": "user", "content": "hello"}]})
        assert response.status_code == 503
