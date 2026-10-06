"""Translate deployment JSON to vLLM CLI arguments without shell interpolation."""
import json
import math
import re


def vllm_arguments(options):
    if not isinstance(options, dict):
        raise ValueError("launch.vllm_args must be an object")
    arguments, seen = [], set()
    for key, value in options.items():
        if not isinstance(key, str) or not re.fullmatch(r"[a-z][a-z0-9_-]*", key):
            raise ValueError("vLLM argument names must be lowercase without leading --")
        name = key.replace("_", "-")
        if name in seen:
            raise ValueError(f"Duplicate vLLM argument: {name}")
        seen.add(name)
        if name in {"model", "served-model-name", "host", "port"}:
            raise ValueError(f"Configure {name} in replica/start_command, not vllm_args")
        if value is None:
            continue  # Leave the installed backend's default unchanged.
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"Nonfinite vLLM argument: {name}")
        if name in {"max-model-len", "max-num-seqs", "max-num-batched-tokens",
                    "tensor-parallel-size", "pipeline-parallel-size", "block-size"}:
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if name == "gpu-memory-utilization":
            if type(value) not in (int, float) or not 0 < value <= 1:
                raise ValueError("gpu-memory-utilization must be in (0, 1]")
        if isinstance(value, bool):
            arguments.append("--" + (name if value else "no-" + name))
        elif isinstance(value, (str, int, float)):
            arguments.extend(("--" + name, str(value)))
        elif isinstance(value, dict):
            try:
                encoded = json.dumps(value, allow_nan=False, separators=(",", ":"))
            except (ValueError, TypeError) as exc:
                raise ValueError(f"Invalid JSON vLLM argument: {name}") from exc
            arguments.extend(("--" + name, encoded))
        elif isinstance(value, list) and value and all(type(v) in (str, int, float) for v in value):
            if any(isinstance(v, float) and not math.isfinite(v) for v in value):
                raise ValueError(f"Nonfinite vLLM argument: {name}")
            arguments.extend(("--" + name, *(str(v) for v in value)))
        else:
            raise ValueError(f"Unsupported vLLM argument value: {name}")
    return arguments


def validate_vllm_command(command, options):
    vllm_arguments(options)
    count = sum(part.count("{vllm_args}") for part in command)
    if options and count != 1:
        raise ValueError("start_command must contain exactly one {vllm_args} placeholder")
    if count > 1:
        raise ValueError("Duplicate {vllm_args} placeholder")
    for index, part in enumerate(command):
        if "{vllm_args}" in part and part != "{vllm_args}":
            # Only an explicit shell command may contain an inline argument list.
            if index < 2 or command[index - 1] not in {"-c", "-lc"} or command[index - 2] not in {"bash", "sh", "/bin/bash", "/bin/sh"}:
                raise ValueError("Inline {vllm_args} requires a bash/sh -c/-lc command")
    flags = set(re.findall(r"(?:^|\s)--([a-z][a-z0-9-]*)(?=\s|=|$)", " ".join(command)))
    for key, value in options.items():
        name = key.replace("_", "-")
        if value is not None and (name in flags or "no-" + name in flags):
            raise ValueError(f"vLLM argument appears in both start_command and vllm_args: {name}")
