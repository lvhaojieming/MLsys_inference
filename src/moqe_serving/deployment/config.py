"""Preparation settings for nodes with an existing, uniform runtime environment."""
from dataclasses import dataclass, field
from pathlib import PurePosixPath
import math


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
