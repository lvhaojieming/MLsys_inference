"""Inference adapter for the existing full-token router architecture."""
import sys
import threading
import time
from collections.abc import Mapping
from pathlib import Path

from ..config import RouterSettings
from .runtime import RoutingDecision


class AscendRouterRuntime:
    def __init__(self, settings: RouterSettings):
        source = Path(settings.training_code) / "src"
        if not (source / "moqe_router/model.py").is_file():
            raise ValueError("training_code must contain the matching moqe_router architecture")
        sys.path.insert(0, str(source))
        import torch
        import torch_npu  # Registers the NPU backend.
        from transformers import AutoTokenizer
        from moqe_router.evaluation import load_router

        if not settings.device.startswith("npu:"):
            raise ValueError("Ascend runtime requires an npu device")
        self.torch = torch
        self.device = torch.device(settings.device)
        torch.npu.set_device(self.device)
        torch.backends.mha.set_fastpath_enabled(False)
        checkpoint = torch.load(settings.checkpoint, map_location="cpu", weights_only=True)
        self.max_prompt_tokens = checkpoint["training_config"]["max_prompt_tokens"]
        ids = tuple(checkpoint["architecture_config"]["expert_ids"])
        if set(settings.expert_mapping) != set(ids):
            raise ValueError("Expert mapping must match every checkpoint expert id exactly")
        if len(set(settings.expert_mapping.values())) != len(ids):
            raise ValueError("Each checkpoint output must map to a distinct expert pool")
        self.experts = tuple(settings.expert_mapping[id_] for id_ in ids)
        del checkpoint
        self.router, self.embedding, _ = load_router(settings.checkpoint, self.device)
        self.router.requires_grad_(False)
        self.tokenizer = AutoTokenizer.from_pretrained(settings.tokenizer, local_files_only=True)
        self.chat_template = self.tokenizer.get_chat_template()
        if not self.chat_template:
            raise ValueError("Canonical tokenizer must supply a chat template")
        self.lock = threading.Lock()
        self.route([{"role": "user", "content": "Hello"}], 128, {"enable_thinking": False})

    def route(self, messages, max_new_tokens, template_kwargs):
        with self.lock:
            torch = self.torch
            torch.npu.set_device(self.device)  # Device context is thread-local.
            started = time.perf_counter()
            ids = self.tokenizer.apply_chat_template(
                messages, chat_template=self.chat_template, tokenize=True,
                add_generation_prompt=True, return_dict=False, **template_kwargs,
            )
            if isinstance(ids, Mapping):
                ids = ids["input_ids"]
            if ids and isinstance(ids[0], list):
                ids = ids[0]
            if not ids or len(ids) > self.max_prompt_tokens:
                raise ValueError(f"Prompt must have 1..{self.max_prompt_tokens} tokens; no truncation")
            tokens = torch.tensor([ids], dtype=torch.long, device=self.device)
            mask = torch.ones_like(tokens, dtype=torch.bool)
            generation = torch.tensor([max_new_tokens], dtype=torch.long, device=self.device)
            with torch.inference_mode(), torch.autocast("npu", dtype=torch.bfloat16):
                logits = self.router(self.embedding(tokens), mask, generation)
                if not bool(torch.isfinite(logits).all().cpu()):
                    raise RuntimeError("Router produced nonfinite logits")
                probabilities = logits.float().softmax(-1).cpu()[0].tolist()
            index = max(range(len(probabilities)), key=probabilities.__getitem__)
            return RoutingDecision(self.experts[index], dict(zip(self.experts, probabilities)),
                                   len(ids), (time.perf_counter() - started) * 1000)
