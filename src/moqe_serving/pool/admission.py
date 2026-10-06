"""Probe advertised model, real answer, and complete streaming before admission."""
import asyncio
import json
import os
import time
import logging
import httpx
import anyio

from ..backends.openai import build_request
from ..config import Admission


async def validate_replica(registry, replica, client, timeout, options=None):
    options = options or Admission()
    registry.begin_validation(replica.id)
    start = time.perf_counter()
    result = {"started_at_unix": time.time(), "checks": []}
    try:
        with anyio.fail_after(timeout):
            headers = {}
            if replica.api_key_env:
                key = os.environ.get(replica.api_key_env)
                if not key:
                    raise ValueError("Missing backend API key")
                headers["Authorization"] = f"Bearer {key}"
            while True:
                try:
                    response = await client.get(replica.base_url.rstrip("/") + "/models", headers=headers)
                    response.raise_for_status()
                    break
                except httpx.HTTPError:
                    await asyncio.sleep(options.poll_interval_seconds)
            if replica.model not in {m.get("id") for m in response.json().get("data", [])}:
                raise ValueError("Configured model not advertised by backend")
            result["model_ready_seconds"] = time.perf_counter() - start
            result["checks"].append("model_identity")
            payload = {"messages": [{"role": "user", "content": options.prompt}],
                       "temperature": options.temperature, "max_tokens": options.max_tokens,
                       "chat_template_kwargs": {"enable_thinking": options.enable_thinking}}
            response = await client.send(build_request(client, replica, {**payload, "stream": False}))
            response.raise_for_status()
            if response.json()["choices"][0]["message"]["content"].strip() != options.expected.strip():
                raise ValueError("Non-streaming correctness check failed")
            result["checks"].append("generation_correctness")
            response = await client.send(build_request(client, replica, {**payload, "stream": True}), stream=True)
            content, done = [], False
            try:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        done = True
                        break
                    event = json.loads(data)
                    if event.get("error"):
                        raise ValueError("Backend stream error")
                    for choice in event.get("choices", []):
                        content.append(choice.get("delta", {}).get("content") or "")
            finally:
                await response.aclose()
            if not done or "".join(content).strip() != options.expected.strip():
                raise ValueError("Streaming correctness/completion check failed")
            result["checks"].append("streaming_correctness")
        result["passed"] = True
    except Exception as exc:
        result["passed"] = False
        result["error"] = str(exc) or type(exc).__name__
    finally:
        result.setdefault("passed", False)
        if not result["passed"]:
            result.setdefault("error", "Validation interrupted")
        registry.finish_validation(replica.id, result["passed"])
        result["total_seconds"] = time.perf_counter() - start
        registry.validation[replica.id] = result
        logging.getLogger(__name__).info("admission id=%s state=%s result=%s",
                                         replica.id, registry.states[replica.id], json.dumps(result))
    return result
