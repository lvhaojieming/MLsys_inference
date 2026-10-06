import json
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse


@dataclass(frozen=True)
class Node:
    id: str
    host: str
    ssh_target: str | None = None
    container: str | None = None
    ssh_options: tuple[str, ...] = ("-o", "BatchMode=yes")
    enabled: bool = True


@dataclass(frozen=True)
class Launch:
    # Commands must start a detached service and return. Stop commands stop only that service.
    start_command: tuple[str, ...] = ()
    stop_command: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)
    command_timeout_seconds: float = 60.0


@dataclass(frozen=True)
class Admission:
    prompt: str = "What is 17 + 25? Reply with the number only."
    expected: str = "42"
    max_tokens: int = 32
    temperature: float = 0.0
    enable_thinking: bool = False
    poll_interval_seconds: float = 0.5


@dataclass(frozen=True)
class Gateway:
    host: str = "127.0.0.1"
    port: int = 8000
    log_level: str = "info"


@dataclass(frozen=True)
class Health:
    enabled: bool = True
    interval_seconds: float = 2.0
    probe_timeout_seconds: float = 2.0
    failure_threshold: int = 3
    recovery_threshold: int = 2


@dataclass(frozen=True)
class Replica:
    id: str
    expert: str
    base_url: str
    model: str
    api_key_env: str | None = None
    enabled: bool = True
    node_id: str | None = None
    device_ids: tuple[int, ...] = ()
    model_path: str | None = None
    backend_port: int | None = None
    launch: Launch | None = None


@dataclass(frozen=True)
class RouterSettings:
    checkpoint: str
    tokenizer: str
    training_code: str
    expert_mapping: dict[str, str]
    device: str = "npu:0"
    embedding_graph: bool = False
    graph_buckets: tuple[int, ...] = (64, 128, 256, 512, 1024)
    graph_threshold_margin: float = 0.01
    embedding_model: str | None = None

    def __post_init__(self):
        if type(self.embedding_graph) is not bool:
            raise ValueError("embedding_graph must be boolean")
        if not self.graph_buckets or any(type(n) is not int or n < 2 for n in self.graph_buckets):
            raise ValueError("graph_buckets must contain positive token lengths >= 2")
        if list(self.graph_buckets) != sorted(set(self.graph_buckets)):
            raise ValueError("graph_buckets must be strictly increasing")
        if not 0 <= self.graph_threshold_margin < 0.5:
            raise ValueError("graph_threshold_margin must be in [0, 0.5)")
        if self.embedding_model is not None and (not isinstance(self.embedding_model, str) or not self.embedding_model.strip()):
            raise ValueError("embedding_model must be a nonempty model directory")


@dataclass(frozen=True)
class Settings:
    replicas: tuple[Replica, ...] = ()
    timeout_seconds: float = 120.0
    max_connections: int = 256
    router: RouterSettings | None = None
    admission_enabled: bool = True
    admin_token_env: str | None = None
    admission_timeout_seconds: float = 300.0
    nodes: tuple[Node, ...] = ()
    admission: Admission = field(default_factory=Admission)
    gateway: Gateway = field(default_factory=Gateway)
    health: Health = field(default_factory=Health)
    watch_config: bool = False
    config_poll_interval_seconds: float = 2.0
    drain_timeout_seconds: float = 300.0
    lifecycle_log_dir: str = "logs/lifecycle"
    default_max_tokens: int = 128
    default_enable_thinking: bool = False

    def __post_init__(self):
        if self.replicas and not self.admission_enabled:
            raise ValueError("Configured instances must pass admission before becoming READY")
        if self.timeout_seconds <= 0 or self.max_connections <= 0:
            raise ValueError("timeout_seconds and max_connections must be positive")
        if self.admission_timeout_seconds <= 0:
            raise ValueError("admission_timeout_seconds must be positive")
        if self.config_poll_interval_seconds <= 0 or self.drain_timeout_seconds <= 0:
            raise ValueError("Config polling and drain timeouts must be positive")
        if self.watch_config and not self.admission_enabled:
            raise ValueError("Config watching requires admission_enabled")
        if not 1 <= self.gateway.port <= 65535:
            raise ValueError("Invalid gateway port")
        if self.gateway.log_level not in {"debug", "info", "warning", "error", "critical"}:
            raise ValueError("Invalid gateway log_level")
        if min(self.health.interval_seconds, self.health.probe_timeout_seconds,
               self.health.failure_threshold, self.health.recovery_threshold) <= 0:
            raise ValueError("Health intervals and thresholds must be positive")
        if self.admission.max_tokens <= 0 or self.admission.poll_interval_seconds <= 0:
            raise ValueError("Invalid admission generation or polling parameters")
        if self.default_max_tokens <= 0:
            raise ValueError("default_max_tokens must be positive")
        assigned_devices = set()
        assigned_endpoints = set()
        for replica in self.replicas:
            node = next((n for n in self.nodes if n.id == replica.node_id), None)
            if not replica.enabled or (node and not node.enabled):
                continue
            if replica.base_url.rstrip('/') in assigned_endpoints:
                raise ValueError("Enabled replicas must have distinct endpoints")
            assigned_endpoints.add(replica.base_url.rstrip('/'))
            for device in replica.device_ids:
                assignment = (replica.node_id, device)
                if replica.node_id and assignment in assigned_devices:
                    raise ValueError("A node/device may only belong to one enabled replica")
                assigned_devices.add(assignment)
        node_ids = [n.id for n in self.nodes]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("Node ids must be unique")
        if any(not n.id or not n.host or not isinstance(n.enabled, bool) for n in self.nodes):
            raise ValueError("Nodes require id, host and a boolean enabled flag")
        ids = [r.id for r in self.replicas]
        if len(ids) != len(set(ids)):
            raise ValueError("Replica ids must be unique")
        for replica in self.replicas:
            if not isinstance(replica.enabled, bool):
                raise ValueError("Replica enabled must be boolean")
            if replica.node_id and replica.node_id not in node_ids:
                raise ValueError(f"Unknown node for {replica.id}")
            if any(isinstance(d, bool) or not isinstance(d, int) or d < 0 for d in replica.device_ids):
                raise ValueError("device_ids must be nonnegative integers")
            if replica.backend_port is not None and not 1 <= replica.backend_port <= 65535:
                raise ValueError("Invalid backend port")
            if replica.backend_port is not None and urlparse(replica.base_url).port != replica.backend_port:
                raise ValueError("backend_port must match base_url port")
            if replica.launch:
                if not replica.launch.start_command or not replica.launch.stop_command:
                    raise ValueError("Managed replica requires both start_command and stop_command")
                if replica.launch.command_timeout_seconds <= 0:
                    raise ValueError("Command timeout must be positive")
                if not replica.model_path or not replica.device_ids:
                    raise ValueError("Managed replica requires model_path and device_ids")
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
        replicas = []
        for item in value.get("replicas", []):
            item = dict(item)
            item["device_ids"] = tuple(item.get("device_ids", []))
            if item.get("launch") is not None:
                launch = dict(item["launch"])
                for key in ("start_command", "stop_command"):
                    launch[key] = tuple(launch.get(key, []))
                item["launch"] = Launch(**launch)
            replicas.append(Replica(**item))
        value["replicas"] = tuple(replicas)
        value["nodes"] = tuple(Node(**{**n, "ssh_options": tuple(n.get("ssh_options", ("-o", "BatchMode=yes")))})
                               for n in value.get("nodes", []))
        for key, kind in (("admission", Admission), ("gateway", Gateway), ("health", Health)):
            if key in value:
                value[key] = kind(**value[key])
        if value.get("router") is not None:
            if "graph_buckets" in value["router"]:
                value["router"]["graph_buckets"] = tuple(value["router"]["graph_buckets"])
            value["router"] = RouterSettings(**value["router"])
        return cls(**value)

    @property
    def active_replicas(self):
        disabled_nodes = {node.id for node in self.nodes if not node.enabled}
        return tuple(r for r in self.replicas if r.enabled and r.node_id not in disabled_nodes)
