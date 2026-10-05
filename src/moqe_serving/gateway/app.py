import logging
import uuid
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from ..backends.openai import build_request
from ..config import Settings
from ..pool.registry import Registry

logger = logging.getLogger(__name__)


def create_app(settings: Settings, transport=None):
    registry = Registry(settings.replicas)

    @asynccontextmanager
    async def lifespan(app):
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
        return {"status": "ok", "configured_replicas": len(registry.replicas)}

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [
            {"id": expert, "object": "model", "owned_by": "moqe"}
            for expert in registry.experts
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
        if not isinstance(expert, str) or expert not in registry.experts:
            raise HTTPException(400, "model must name a configured expert; automatic routing is pending")
        if not isinstance(payload.get("messages"), list) or not payload["messages"]:
            raise HTTPException(400, "messages must be a nonempty list")
        if not isinstance(payload.get("stream", False), bool):
            raise HTTPException(400, "stream must be boolean")

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
        if payload.get("stream", False):
            async def events():
                try:
                    async for chunk in upstream.aiter_bytes():
                        yield chunk
                finally:
                    await upstream.aclose()
                    registry.release(replica)
            return StreamingResponse(events(), media_type="text/event-stream", headers=headers)

        try:
            await upstream.aread()
            result = upstream.json()
            if not isinstance(result, dict):
                raise ValueError("Invalid backend response")
            result["model"] = expert
            return JSONResponse(result, headers=headers)
        except (ValueError, httpx.HTTPError) as exc:
            raise HTTPException(502, "Invalid or interrupted backend response") from exc
        finally:
            await upstream.aclose()
            registry.release(replica)

    return app
