"""Preparation settings for nodes with an existing, uniform runtime environment."""
from dataclasses import dataclass, field, replace
from pathlib import PurePosixPath
import math
import re

from ..vllm_options import vllm_arguments


@dataclass(frozen=True)
class NodePreparation:
    python: str = "python3"
    required_modules: tuple[str, ...] = ("vllm",)
    required_paths: tuple[str, ...] = ()
    host_checks: tuple[tuple[str, ...], ...] = ()
    runtime_checks: tuple[tuple[str, ...], ...] = ()
    env: dict[str, str] = field(default_factory=dict)
    environment_scripts: tuple[str, ...] = ()
    helper_path: str = "/workspace/moqe/manage_backend.py"
    command_timeout_seconds: float = 120.0

    def __post_init__(self):
        if (not isinstance(self.python, str) or not self.python
                or not isinstance(self.helper_path, str)
                or not PurePosixPath(self.helper_path).is_absolute()):
            raise ValueError("Node preparation requires python and an absolute helper_path")
        if (type(self.command_timeout_seconds) not in (int, float)
                or not math.isfinite(self.command_timeout_seconds)
                or self.command_timeout_seconds <= 0):
            raise ValueError("Node preparation timeout must be positive")
        if any(not isinstance(v, str) or not v for v in (*self.required_modules, *self.required_paths)):
            raise ValueError("Preparation modules and paths must be nonempty strings")
        if any(not isinstance(v, str) or not PurePosixPath(v).is_absolute() for v in self.environment_scripts):
            raise ValueError("Preparation environment_scripts must be absolute paths")
        for command in (*self.host_checks, *self.runtime_checks):
            if not command or any(not isinstance(v, str) or not v for v in command):
                raise ValueError("Preparation checks must be nonempty command arrays")
        if any(not isinstance(k, str) or not isinstance(v, str) for k, v in self.env.items()):
            raise ValueError("Preparation environment must contain strings")

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        for key in ("required_modules", "required_paths", "environment_scripts"):
            if key in value:
                if not isinstance(value[key], list):
                    raise ValueError(f"prepare.{key} must be an array")
                value[key] = tuple(value[key])
        for key in ("host_checks", "runtime_checks"):
            if key in value:
                if not isinstance(value[key], list) or any(not isinstance(c, list) for c in value[key]):
                    raise ValueError(f"prepare.{key} must contain command arrays")
                value[key] = tuple(tuple(c) for c in value[key])
        return cls(**value)


@dataclass(frozen=True)
class RuntimeProfile:
    """Reusable backend defaults, independent of the logical expert pool."""
    accelerator: str = "ascend"
    backend: str = "vllm"
    adapter: str = "native"
    device_env: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    environment_scripts: tuple[str, ...] = ()
    required_modules: tuple[str, ...] = ()
    host_checks: tuple[tuple[str, ...], ...] | None = None
    runtime_checks: tuple[tuple[str, ...], ...] = ()
    vllm_args: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.accelerator != "ascend" or self.backend != "vllm":
            raise ValueError("Runtime profiles currently support only Ascend with the vllm backend")
        if not isinstance(self.adapter, str) or not self.adapter.strip():
            raise ValueError("Runtime adapter must be a nonempty name")
        if self.device_env is not None and self.device_env != self.visibility_variable:
            raise ValueError("device_env must match the profile accelerator")
        # Reuse the same preparation validation rather than maintaining two schemas.
        NodePreparation(env=self.env, environment_scripts=self.environment_scripts,
            required_modules=self.required_modules, host_checks=self.host_checks or (),
            runtime_checks=self.runtime_checks)
        if any(not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key) for key in self.env):
            raise ValueError("Invalid runtime profile environment variable")
        vllm_arguments(self.vllm_args)

    @property
    def visibility_variable(self):
        return "ASCEND_RT_VISIBLE_DEVICES"

    @property
    def backend_module(self):
        return "vllm.entrypoints.openai.api_server"

    @property
    def backend_arguments(self):
        defaults = {} if self.adapter == "native" else {"quantization": self.adapter}
        return {**defaults, **self.vllm_args}

    def preparation(self, base, replicas):
        base = base or NodePreparation()
        modules = ("vllm", "torch_npu", "vllm_ascend")
        checks = self.host_checks
        if checks is None:
            checks = (("npu-smi", "info"),)
        devices = ",".join(map(str, sorted({d for r in replicas for d in r.device_ids})))
        return replace(base,
            env={**self.env, **base.env, self.visibility_variable: devices},
            environment_scripts=tuple(dict.fromkeys((*self.environment_scripts, *base.environment_scripts))),
            required_modules=tuple(dict.fromkeys((*modules, *self.required_modules, *base.required_modules))),
            host_checks=tuple(dict.fromkeys((*checks, *base.host_checks))),
            runtime_checks=tuple(dict.fromkeys((*self.runtime_checks, *base.runtime_checks))))

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        for key in ("environment_scripts", "required_modules"):
            if key in value:
                if not isinstance(value[key], list):
                    raise ValueError(f"runtime_profiles.{key} must be an array")
                value[key] = tuple(value[key])
        for key in ("host_checks", "runtime_checks"):
            if key in value:
                if key == "host_checks" and value[key] is None:
                    continue
                if not isinstance(value[key], list) or any(not isinstance(c, list) for c in value[key]):
                    raise ValueError(f"runtime_profiles.{key} must contain command arrays")
                value[key] = tuple(tuple(c) for c in value[key])
        return cls(**value)
