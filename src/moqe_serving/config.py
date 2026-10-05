import json
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse


@dataclass(frozen=True)
class Replica:
    id: str
    expert: str
    base_url: str
    model: str
    api_key_env: str | None = None


@dataclass(frozen=True)
class RouterSettings:
    checkpoint: str
    tokenizer: str
    training_code: str
    expert_mapping: dict[str, str]
    device: str = "npu:0"


@dataclass(frozen=True)
class Settings:
    replicas: tuple[Replica, ...] = ()
    timeout_seconds: float = 120.0
    max_connections: int = 256
    router: RouterSettings | None = None

    def __post_init__(self):
        if self.timeout_seconds <= 0 or self.max_connections <= 0:
            raise ValueError("timeout_seconds and max_connections must be positive")
        ids = [r.id for r in self.replicas]
        if len(ids) != len(set(ids)):
            raise ValueError("Replica ids must be unique")
        for replica in self.replicas:
            if replica.expert == "auto":
                raise ValueError("auto is reserved for learned routing")
            url = urlparse(replica.base_url)
            if not replica.id or not replica.expert or not replica.model:
                raise ValueError("Replica id, expert and model are required")
            if url.scheme not in {"http", "https"} or not url.netloc or url.query or url.fragment:
                raise ValueError(f"Invalid backend URL for {replica.id}")
            if not url.path.rstrip("/").endswith("/v1"):
                raise ValueError("Backend base_url must end in /v1")
        if self.router:
            experts = {r.expert for r in self.replicas}
            if not self.router.expert_mapping or not set(self.router.expert_mapping.values()) <= experts:
                raise ValueError("Router mapping must reference configured expert pools")

    @classmethod
    def load(cls, path: str):
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        value["replicas"] = tuple(Replica(**r) for r in value.get("replicas", []))
        if value.get("router") is not None:
            value["router"] = RouterSettings(**value["router"])
        return cls(**value)
