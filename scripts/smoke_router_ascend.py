"""Load the existing trained router on one NPU; no training or expert calls."""
import argparse
from collections.abc import Mapping
import hashlib
import json
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-code", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(args.training_code) / "src"))
    import torch
    import torch_npu
    from transformers import AutoTokenizer
    from moqe_router.evaluation import load_router

    torch.npu.set_device(0)
    torch.backends.mha.set_fastpath_enabled(False)
    device = torch.device("npu:0")
    checkpoint_path = Path(args.checkpoint)
    checkpoint_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    maximum = checkpoint["training_config"]["max_prompt_tokens"]
    metadata = {k: checkpoint.get(k) for k in ("epoch", "global_step")}
    del checkpoint
    start = time.monotonic()
    router, embedding, architecture = load_router(args.checkpoint, device)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    torch.npu.synchronize()
    report = {
        "checkpoint": args.checkpoint, "checkpoint_sha256": checkpoint_hash,
        "checkpoint_metadata": metadata, "physical_npu": 1, "logical_npu": 0,
        "precision": "bf16_autocast", "load_seconds": time.monotonic() - start,
        "expert_ids": list(architecture.expert_ids), "max_prompt_tokens": maximum,
        "router_parameters": sum(p.numel() for p in router.parameters()),
        "embedding_parameters": embedding.weight.numel(),
        "embedding_dtype": str(embedding.weight.dtype), "cases": [],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    def run(name, tokens):
        if len(tokens) > maximum:
            raise ValueError("Prompt exceeds trained context; truncation is forbidden")
        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        mask = torch.ones_like(ids, dtype=torch.bool)
        lengths = torch.tensor([128], dtype=torch.long, device=device)
        def forward():
            with torch.inference_mode(), torch.autocast("npu", dtype=torch.bfloat16):
                return router(embedding(ids), mask, lengths)
        forward()
        torch.npu.synchronize()
        torch.npu.reset_peak_memory_stats()
        samples = []
        for _ in range(5):
            before = time.perf_counter()
            logits = forward()
            torch.npu.synchronize()
            samples.append(1000 * (time.perf_counter() - before))
        probabilities = logits.float().softmax(-1).cpu()[0].tolist()
        selected = max(range(len(probabilities)), key=probabilities.__getitem__)
        result = {
            "name": name, "input_tokens": len(tokens), "processed_tokens": int(mask.sum().cpu()),
            "probabilities": probabilities, "selected_expert": architecture.expert_ids[selected],
            "forward_ms_median": sorted(samples)[len(samples) // 2], "forward_ms_samples": samples,
            "allocated_gib": torch.npu.memory_allocated() / 2**30,
            "peak_allocated_gib": torch.npu.max_memory_allocated() / 2**30,
            "reserved_gib": torch.npu.memory_reserved() / 2**30,
            "finite": bool(torch.isfinite(logits).all().cpu()),
        }
        report["cases"].append(result)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False), flush=True)

    for prompt in ("What is 17 + 25? Reply with the number only.",
                   "请用一句简短的中文介绍北京。",
                   "A shop sells 12 boxes with 13 pencils each. How many pencils are there?"):
        tokens = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                              tokenize=True, add_generation_prompt=True,
                                              enable_thinking=False, return_dict=False)
        if isinstance(tokens, Mapping):
            tokens = tokens["input_ids"]
        if tokens and isinstance(tokens[0], list):
            tokens = tokens[0]
        run(prompt, tokens)
    # Synthetic lengths test memory/context support, not routing quality.
    base = tokenizer.encode("This is a router context length smoke test. ", add_special_tokens=False)
    for length in (128, 512, 2048, 8192, maximum):
        run(f"synthetic_{length}", (base * ((length + len(base) - 1) // len(base)))[:length])
    report["checkpoint_unchanged"] = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() == checkpoint_hash
    report["all_tokens_processed"] = all(c["processed_tokens"] == c["input_tokens"] for c in report["cases"])
    report["passed"] = report["checkpoint_unchanged"] and report["all_tokens_processed"] and all(c["finite"] for c in report["cases"])
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("ROUTER_SMOKE_PASS", report["passed"], flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
