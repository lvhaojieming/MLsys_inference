import os
import httpx

from ..config import Replica


def build_request(client: httpx.AsyncClient, replica: Replica, payload: dict):
    headers = {}
    if replica.api_key_env:
        key = os.environ.get(replica.api_key_env)
        if not key:
            raise ValueError(f"Missing backend API key environment variable: {replica.api_key_env}")
        headers["Authorization"] = f"Bearer {key}"
    return client.build_request(
        "POST", replica.base_url.rstrip("/") + "/chat/completions",
        json={**payload, "model": replica.model}, headers=headers,
    )
