import logging
import asyncio
import json
import uuid
import os
import secrets
from contextlib import asynccontextmanager

import httpx
import anyio
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from ..backends.openai import build_request
from ..config import Replica, Settings
from ..pool.registry import Registry
from ..pool.admission import validate_replica
from ..pool.controller import ConfigController
from ..pool.health import HealthManager

logger = logging.getLogger(__name__)


def create_app(settings: Settings, transport=None, router_runtime=None, config_path=None):
    controlled = settings.watch_config or any(r.launch for r in settings.replicas)
    if settings.watch_config and not config_path:
        raise ValueError("Config watching requires a configuration file path")
    registry = Registry(() if controlled else settings.active_replicas,
                        require_admission=settings.admission_enabled)

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
            app.state.controller = ConfigController(settings, registry, client, config_path)
            watcher = None
            health_task = None
            if controlled:
                await app.state.controller.reconcile(settings)
                if settings.watch_config:
                    watcher = asyncio.create_task(app.state.controller.watch())
            elif settings.admission_enabled:
                await asyncio.gather(*(validate_replica(registry, r, client, settings.admission_timeout_seconds, settings.admission)
                                       for r in registry.replicas))
            app.state.health_manager = HealthManager(settings, registry, client)
            if settings.health.enabled:
                health_task = asyncio.create_task(app.state.health_manager.run())
            try:
                yield
            finally:
                for task in (watcher, health_task):
                    if not task:
                        continue
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

    app = FastAPI(title="MoQE Inference Gateway", lifespan=lifespan)
    app.state.registry = registry

    def require_admin(request):
        token = os.environ.get(settings.admin_token_env, "") if settings.admin_token_env else ""
        if not token:
            raise HTTPException(404, "Management API is disabled")
        if not secrets.compare_digest(request.headers.get("Authorization", ""), "Bearer " + token):
            raise HTTPException(401, "Invalid management token")

    @app.get("/admin/instances")
    async def instances(request: Request):
        require_admin(request)
        return {"instances": registry.snapshot()}

    @app.get("/admin/config")
    async def config_status(request: Request):
        require_admin(request)
        return app.state.controller.snapshot()

    @app.post("/admin/config/reload")
    async def reload_config(request: Request):
        require_admin(request)
        if not settings.watch_config:
            raise HTTPException(409, "Enable watch_config to reconcile the configuration")
        try:
            await app.state.controller.reload()
        except (ValueError, TypeError, OSError) as exc:
            app.state.controller.last_error = str(exc)
            raise HTTPException(400, str(exc)) from exc
        except (RuntimeError, TimeoutError) as exc:
            app.state.controller.last_error = str(exc)
            raise HTTPException(409, str(exc)) from exc
        return app.state.controller.snapshot()

    @app.post("/admin/instances")
    async def register(request: Request):
        require_admin(request)
        if controlled:
            raise HTTPException(409, "Edit replicas in the source configuration instead")
        try:
            replica = Replica(**await request.json())
            # Reuse configuration validation for dynamically supplied endpoints.
            if replica.launch:
                raise ValueError("Managed replicas must be configured in the source file")
            Settings(replicas=(replica,), nodes=settings.nodes)
        except (ValueError, TypeError):
            raise HTTPException(400, "Invalid replica configuration")
        if replica.expert not in registry.experts:
            raise HTTPException(400, "Only replicas of an existing expert pool may be added")
        if any(n.id == replica.node_id and not n.enabled for n in settings.nodes):
            raise HTTPException(409, "The replica's node is disabled in configuration")
        if replica.model not in {r.model for r in registry.replicas if r.expert == replica.expert}:
            raise HTTPException(400, "New replica must use the existing expert's served model name")
        try:
            registry.add(replica)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        result = await validate_replica(registry, replica, app.state.client, settings.admission_timeout_seconds, settings.admission)
        auto_eligible = bool(settings.router and replica.expert in settings.router.expert_mapping.values())
        return JSONResponse({"id": replica.id, "state": registry.states[replica.id],
                             "auto_routing_eligible": auto_eligible, "validation": result},
                            status_code=201 if registry.states[replica.id] == "ready" else 422)

    @app.post("/admin/instances/{replica_id}/validate")
    async def revalidate(replica_id: str, request: Request):
        require_admin(request)
        if controlled:
            raise HTTPException(409, "Disable and re-enable the replica in the source configuration")
        try:
            replica = registry.get(replica_id)
        except StopIteration:
            raise HTTPException(404, "Unknown replica")
        try:
            result = await validate_replica(registry, replica, app.state.client, settings.admission_timeout_seconds, settings.admission)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        return JSONResponse({"state": registry.states[replica_id], "validation": result},
                            status_code=200 if registry.states[replica_id] == "ready" else 422)

    @app.post("/admin/instances/{replica_id}/drain")
    async def drain(replica_id: str, request: Request):
        require_admin(request)
        if controlled:
            raise HTTPException(409, "Disable or remove the replica in the source configuration")
        try:
            registry.drain(replica_id)
        except KeyError:
            raise HTTPException(404, "Unknown replica")
        return {"id": replica_id, "state": registry.states[replica_id]}

    @app.get("/ready")
    async def ready():
        experts = sorted({r.expert for r in registry.replicas if registry.states[r.id] == "ready"})
        required = set(settings.router.expert_mapping.values()) if settings.router else set(registry.experts)
        passed = bool(experts) and required <= set(experts)
        return JSONResponse({"ready": passed, "ready_experts": experts}, status_code=200 if passed else 503)

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
            generation = payload.get("max_tokens", settings.default_max_tokens)
            if "max_completion_tokens" in payload or isinstance(generation, bool) or not isinstance(generation, int) or generation <= 0:
                raise HTTPException(400, "Use a positive integer max_tokens for automatic routing")
            kwargs = payload.get("chat_template_kwargs", {"enable_thinking": settings.default_enable_thinking})
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

        try:
            replica = registry.acquire(expert)
        except KeyError:
            raise HTTPException(503, "Expert pool has no admitted ready replica")
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
