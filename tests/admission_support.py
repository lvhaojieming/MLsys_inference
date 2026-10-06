"""Successful startup probes for tests focused on the subsequent proxy requests."""
import json
import httpx


def backend(request):
    """A tiny backend supporting model identity, answers and complete SSE."""
    if request.url.path.endswith('/models'):
        return httpx.Response(200, json={"data": [{"id": "backend"}]})
    payload = json.loads(request.content)
    answer = "0" if request.url.host == "wrong" else "42"
    if payload.get('stream'):
        event = {"choices": [{"delta": {"content": answer}}]}
        return httpx.Response(200, text='data: ' + json.dumps(event) + '\n\ndata: [DONE]\n\n')
    return httpx.Response(200, json={"choices": [{"message": {"content": answer}}]})


def with_admission(handler, models):
    def transport(request):
        if request.url.path.endswith('/models'):
            return httpx.Response(200, json={'data': [{'id': m} for m in models]})
        payload = json.loads(request.content)
        if payload.get('messages', [{}])[0].get('content') == 'What is 17 + 25? Reply with the number only.':
            if payload.get('stream'):
                return httpx.Response(200, text='data: {"choices":[{"delta":{"content":"42"}}]}\n\ndata: [DONE]\n\n')
            return httpx.Response(200, json={'choices': [{'message': {'content': '42'}}]})
        return handler(request)
    return transport
