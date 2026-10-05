"""Check a running learned-router Gateway against the selected real expert."""
import argparse
import json
from pathlib import Path

from smoke_real_backends import send, PROMPTS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--gateway", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    from transformers import AutoTokenizer
    from moqe_serving.config import Settings
    settings = Settings.load(args.config)
    tokenizer = AutoTokenizer.from_pretrained(settings.router.tokenizer, local_files_only=True)
    template = tokenizer.get_chat_template()
    cases = list(PROMPTS) + [("long_prompt", "This is a router context length smoke test. " * 64
                             + "What is 17 + 25? Reply with the number only.", "42")]
    report = {"gateway": args.gateway, "checkpoint": settings.router.checkpoint, "cases": []}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for name, prompt, expected in cases:
        messages = [{"role": "user", "content": prompt}]
        encoded = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                enable_thinking=False, return_dict=False)
        if hasattr(encoded, "keys"):
            encoded = encoded["input_ids"]
        for stream in (False, True):
            payload = {"model": "auto", "messages": messages, "max_tokens": 128,
                       "temperature": 0, "seed": 42, "stream": stream,
                       "chat_template_kwargs": {"enable_thinking": False}}
            routed = send(args.gateway.rstrip("/") + "/v1", payload)
            headers = {k.lower(): v for k, v in routed.get("headers", {}).items()}
            selected = headers.get("x-moqe-expert")
            replica = next((r for r in settings.replicas if r.expert == selected), None)
            direct = send(replica.base_url, {**payload, "model": replica.model,
                          "chat_template": template}) if replica else {"ok": False, "error": "No selected expert"}
            same = routed.get("result", {}).get("content") == direct.get("result", {}).get("content")
            passed = routed["ok"] and direct["ok"] and same and headers.get("x-moqe-input-tokens") == str(len(encoded))
            if passed and not stream:
                passed = routed["result"]["usage"]["prompt_tokens"] == len(encoded)
            if expected is not None:
                passed = passed and routed.get("result", {}).get("content", "").strip() == expected
            case = {"name": name, "stream": stream, "selected_expert": selected,
                    "full_prompt_tokens": len(encoded), "same_content": same,
                    "pass": passed, "routed": routed, "direct": direct}
            report["cases"].append(case)
            output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            print(name, stream, selected, passed, flush=True)
    report["passed"] = sum(c["pass"] for c in report["cases"])
    report["total"] = len(report["cases"])
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("RESULT", report["passed"], "/", report["total"], flush=True)
    return 0 if report["passed"] == report["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
