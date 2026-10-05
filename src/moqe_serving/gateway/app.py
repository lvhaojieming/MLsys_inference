import logging
import asyncio
import json
import uuid
from contextlib import asynccontextmanager

import httpx
import anyio
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from ..backends.openai import build_request
from ..config import Settings
from ..pool.registry import Registry

logger = logging.getLogger(__name__)


def create_app(settings: Settings, transport=None, router_runtime=None):
    registry = Registry(settings.replicas)

    @asynccontextmanager
    async def lifespan(app):
        app.state.router = router_runtime
        app.state.routing_slot = asyncio.Semaphore(1)
        if settings.router and app.state.router is None:
            from ..routing.ascend import AscendRouterRuntime
            app.state.router = await anyio.to_thread.run_sync(AscendRouterRuntime, settings.router)
        async with httpx.AsyncClient(
            timeout=settings.timeout_seconds,
            limits=httpx.Limits(max_connections=settings.max_connections),
            transport=transport,
        ) as client:
            app.state.client = client
            yield

    app = FastAPI(title="MoQE Inference Gateway", lifespan=lifespan)
    app.state.registry = registry

    @app.get("/health")
    async def health():
        # This reports gateway liveness, not backend health.
        value = {"status": "ok", "configured_replicas": len(registry.replicas)}
        if app.state.router:
            value["router_ready"] = True
        return value

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [
            {"id": expert, "object": "model", "owned_by": "moqe"}
            for expert in registry.experts + (["auto"] if app.state.router else [])
        ]}

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        try:
            payload = await request.json()
        except ValueError:
            raise HTTPException(400, "Request must be valid JSON")
        if not isinstance(payload, dict):
            raise HTTPException(400, "Request must be a JSON object")
        expert = payload.get("model")
        requested_model = expert
        if not isinstance(expert, str) or (expert not in registry.experts and expert != "auto"):
            raise HTTPException(400, "model must name a configured expert or auto")
        if not isinstance(payload.get("messages"), list) or not payload["messages"]:
            raise HTTPException(400, "messages must be a nonempty list")
        if not isinstance(payload.get("stream", False), bool):
            raise HTTPException(400, "stream must be boolean")

        decision = None
        if expert == "auto":
            runtime = app.state.router
            if runtime is None:
                raise HTTPException(503, "Learned router is not configured")
            messages = payload["messages"]
            if any(not isinstance(m, dict) or m.get("role") not in {"system", "user", "assistant"}
                   or not isinstance(m.get("content"), str) for m in messages):
                raise HTTPException(400, "Automatic routing currently requires plain text messages")
            if any(k in payload for k in ("tools", "tool_choice", "chat_template", "documents",
                                         "add_generation_prompt", "continue_final_message",
                                         "truncate_prompt_tokens", "add_special_tokens",
                                         "chat_template_content_format")):
                raise HTTPException(400, "Automatic routing uses the canonical chat template")
            generation = payload.get("max_tokens", 128)
            if "max_completion_tokens" in payload or isinstance(generation, bool) or not isinstance(generation, int) or generation <= 0:
                raise HTTPException(400, "Use a positive integer max_tokens for automatic routing")
            kwargs = payload.get("chat_template_kwargs", {"enable_thinking": False})
            if not isinstance(kwargs, dict) or set(kwargs) - {"enable_thinking"} or not isinstance(kwargs.get("enable_thinking", False), bool):
                raise HTTPException(400, "Only boolean enable_thinking is supported in template kwargs")
            kwargs = {"enable_thinking": kwargs.get("enable_thinking", False)}
            try:
                async with app.state.routing_slot:
                    decision = await anyio.to_thread.run_sync(runtime.route, messages, generation, kwargs)
            except ValueError as exc:
                raise HTTPException(400, str(exc)) from exc
            expert = decision.expert
            if expert not in registry.experts:
                raise HTTPException(503, "Router selected an unavailable expert pool")
            payload = {**payload, "max_tokens": generation, "chat_template": runtime.chat_template,
                       "chat_template_kwargs": kwargs}

        replica = registry.acquire(expert)
        trace_id = uuid.uuid4().hex
        upstream = None
        try:
            upstream = await app.state.client.send(
                build_request(app.state.client, replica, payload), stream=True,
            )
            if upstream.status_code >= 400:
                raise HTTPException(502, f"Backend rejected request (HTTP {upstream.status_code})")
        except (httpx.HTTPError, ValueError) as exc:
            if upstream is not None:
                await upstream.aclose()
            registry.release(replica)
            logger.warning("backend_failure trace=%s replica=%s error=%s", trace_id, replica.id, type(exc).__name__)
            raise HTTPException(502, "Backend unavailable") from exc
        except BaseException:
            if upstream is not None:
                await upstream.aclose()
            registry.release(replica)
            raise

        headers = {"X-Request-ID": trace_id, "X-MoQE-Expert": expert, "X-MoQE-Replica": replica.id}
        if decision:
            headers["X-MoQE-Router-Ms"] = f"{decision.elapsed_ms:.3f}"
            headers["X-MoQE-Input-Tokens"] = str(decision.input_tokens)
            logger.info("route trace=%s expert=%s probabilities=%s input_tokens=%d router_ms=%.3f",
                        trace_id, expert, decision.probabilities, decision.input_tokens, decision.elapsed_ms)
        if payload.get("stream", False):
            async def events():
                try:
                    async for line in upstream.aiter_lines():
                        if line.startswith("data:") and line[5:].strip() != "[DONE]":
                            try:
                                event = json.loads(line[5:])
                                if isinstance(event, dict) and "model" in event:
                                    event["model"] = requested_model
                                    line = "data: " + json.dumps(event, ensure_ascii=False)
                            except ValueError:
                                pass
                        yield (line + "\n").encode("utf-8")
                finally:
                    await upstream.aclose()
                    registry.release(replica)
            return StreamingResponse(events(), media_type="text/event-stream", headers=headers)

        try:
            await upstream.aread()
            result = upstream.json()
            if not isinstance(result, dict):
                raise ValueError("Invalid backend response")
            result["model"] = requested_model
            return JSONResponse(result, headers=headers)
        except (ValueError, httpx.HTTPError) as exc:
            raise HTTPException(502, "Invalid or interrupted backend response") from exc
        finally:
            await upstream.aclose()
            registry.release(replica)

    return app
