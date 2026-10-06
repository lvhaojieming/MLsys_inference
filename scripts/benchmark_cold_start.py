"""Measure fresh service processes, readiness, first content token and warm requests.

The launched command must stay in the foreground. Existing services are never stopped.
OS file caches are retained: these are process-cold, not disk-cold measurements.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.error
import urllib.request


def probe(url):
    with urllib.request.urlopen(url, timeout=2) as response:
        return json.load(response)


def request(base, model, prompt, max_tokens):
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}],
               "max_tokens": max_tokens, "temperature": 0, "seed": 42,
               "stream": True, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(base.rstrip("/") + "/v1/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    first = None
    done = False
    content = []
    with urllib.request.urlopen(req, timeout=180) as response:
        headers = dict(response.headers)
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                done = True
                break
            event = json.loads(data)
            if event.get("error"):
                raise RuntimeError(str(event["error"]))
            for choice in event.get("choices", []):
                text = choice.get("delta", {}).get("content")
                if text:
                    if first is None:
                        first = time.perf_counter() - start
                    content.append(text)
    if first is None or not done:
        raise RuntimeError("Stream missing content or [DONE]")
    return {"ttft_seconds": first, "e2e_seconds": time.perf_counter() - start,
            "content": "".join(content), "headers": headers}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", default="auto")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warm-requests", type=int, default=3)
    parser.add_argument("--ready-timeout", type=float, default=600)
    parser.add_argument("--prompt", default="What is 17 + 25? Reply with the number only.")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--require-router", action="store_true")
    parser.add_argument("--scope", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or args.runs < 1:
        parser.error("A foreground command and positive run count are required")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    report = {"scope": args.scope, "cache_mode": "process_cold_os_cache_retained",
              "command": command, "base_url": args.base_url, "model": args.model,
              "prompt": args.prompt, "max_tokens": args.max_tokens, "runs": []}
    health = args.base_url.rstrip("/") + "/health"
    try:
        probe(health)
    except (urllib.error.URLError, TimeoutError):
        pass
    else:
        raise RuntimeError("Target port already serves health; refusing to reuse it")
    for index in range(args.runs):
        row = {"run": index + 1, "started_at_unix": time.time()}
        with (output / f"startup-{index + 1}.log").open("wb") as log:
            start = time.perf_counter()
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            try:
                while True:
                    if process.poll() is not None:
                        raise RuntimeError(f"Service exited {process.returncode} before ready")
                    if time.perf_counter() - start > args.ready_timeout:
                        raise TimeoutError("Service readiness timeout")
                    try:
                        state = probe(health)
                        if args.require_router and not state.get("router_ready"):
                            raise ValueError("Router not ready")
                        break
                    except (urllib.error.URLError, TimeoutError, ValueError):
                        time.sleep(0.2)
                row["ready_seconds"] = time.perf_counter() - start
                row["first_request"] = request(args.base_url, args.model, args.prompt, args.max_tokens)
                row["startup_to_first_content_seconds"] = time.perf_counter() - start - (
                    row["first_request"]["e2e_seconds"] - row["first_request"]["ttft_seconds"])
                row["warm_requests"] = [request(args.base_url, args.model, args.prompt, args.max_tokens)
                                        for _ in range(args.warm_requests)]
                row["status"] = "ok"
            except Exception as exc:
                row["status"] = "failed"
                row["error"] = str(exc)
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait(timeout=10)
        report["runs"].append(row)
        (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(row), flush=True)
        if row["status"] != "ok":
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
