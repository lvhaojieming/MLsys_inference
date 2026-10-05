"""Real backend smoke check; starts and stops a private Gateway process."""
import argparse
import json
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from moqe_serving.config import Settings

PROMPTS = [
    ("addition", "What is 17 + 25? Reply with the number only.", "42"),
    ("multiplication", "What is 12 times 13? Reply with the number only.", "156"),
    ("chinese", "请用一句简短的中文介绍北京。", None),
]


def send(url, payload):
    started = time.monotonic()
    request = urllib.request.Request(url + "/chat/completions", json.dumps(payload).encode(),
                                     {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            headers = dict(response.headers)
            if payload["stream"]:
                content, reasoning, chunks, done, first, finish = "", "", 0, False, None, None
                for raw in response:
                    line = raw.decode().strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        done = True
                        break
                    value = json.loads(data)
                    if value.get("error"):
                        raise RuntimeError(str(value["error"]))
                    for choice in value.get("choices", []):
                        delta = choice.get("delta", {})
                        text = delta.get("content") or ""
                        thought = delta.get("reasoning_content") or ""
                        if (text or thought) and first is None:
                            first = time.monotonic() - started
                        content += text
                        reasoning += thought
                        chunks += 1
                        finish = choice.get("finish_reason") or finish
                result = {"content": content, "reasoning": reasoning, "chunks": chunks,
                          "done": done, "finish_reason": finish, "first_content_seconds": first}
            else:
                value = json.load(response)
                choice = value["choices"][0]
                result = {"content": choice["message"].get("content") or "",
                          "reasoning": choice["message"].get("reasoning_content"),
                          "finish_reason": choice["finish_reason"], "usage": value.get("usage")}
        return {"ok": bool(result["content"]) and result["finish_reason"] == "stop"
                and result.get("done", True), "seconds": time.monotonic() - started,
                "result": result, "headers": {k: v for k, v in headers.items()
                                                if k.lower().startswith("x-")}}
    except urllib.error.HTTPError as exc:
        return {"ok": False, "status": exc.code, "error": exc.read().decode(),
                "seconds": time.monotonic() - started}
    except Exception as exc:
        return {"ok": False, "error": str(exc), "seconds": time.monotonic() - started}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    settings = Settings.load(args.config)
    if not settings.replicas:
        raise ValueError("Real replicas are required")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    gateway = f"http://127.0.0.1:{port}"
    report = {"config": args.config, "mode": "explicit_expert_no_learned_router", "cases": []}
    with output.with_suffix(".gateway.log").open("w") as logfile:
        process = subprocess.Popen([sys.executable, "-m", "moqe_serving", "--config", args.config,
                                    "--port", str(port)], stdout=logfile, stderr=subprocess.STDOUT)
        try:
            ready = False
            for _ in range(100):
                if process.poll() is not None:
                    raise RuntimeError("Gateway startup failed; see gateway log")
                try:
                    with urllib.request.urlopen(gateway + "/health", timeout=1) as response:
                        report["gateway_health"] = json.load(response)
                    ready = True
                    break
                except (OSError, urllib.error.URLError):
                    time.sleep(0.1)
            if not ready:
                raise RuntimeError("Gateway readiness timeout")
            for replica in settings.replicas:
                for name, prompt, expected in PROMPTS:
                    for stream in [False, True]:
                        payload = {"model": replica.model, "messages": [{"role": "user", "content": prompt}],
                                   "temperature": 0, "seed": 42, "max_tokens": 128, "stream": stream,
                                   "chat_template_kwargs": {"enable_thinking": False}}
                        direct = send(replica.base_url, payload)
                        proxied = send(gateway + "/v1", {**payload, "model": replica.expert})
                        case = {"expert": replica.expert, "replica": replica.id, "prompt": prompt,
                                "name": name, "stream": stream, "direct": direct, "gateway": proxied}
                        case["same_content"] = (direct.get("result", {}).get("content") ==
                                                proxied.get("result", {}).get("content"))
                        case["pass"] = direct["ok"] and proxied["ok"] and case["same_content"]
                        if expected is not None:
                            case["expected"] = expected
                            case["pass"] = case["pass"] and direct["result"]["content"].strip() == expected
                        report["cases"].append(case)
                        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
                        print(replica.expert, name, "stream" if stream else "json", case["pass"], flush=True)
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
    report["passed"] = sum(case["pass"] for case in report["cases"])
    report["total"] = len(report["cases"])
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"RESULT {report['passed']}/{report['total']} {output}")
    return 0 if report["passed"] == report["total"] else 1


if __name__ == "__main__":
    sys.exit(main())
