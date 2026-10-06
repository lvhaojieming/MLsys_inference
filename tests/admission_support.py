"""Successful startup probes for tests focused on the subsequent proxy requests."""
import json
import httpx


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
