from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class RoutingDecision:
    expert: str
    probabilities: dict[str, float]
    input_tokens: int
    elapsed_ms: float


class RouterRuntime(Protocol):
    chat_template: str

    def route(self, messages: list[dict], max_new_tokens: int, template_kwargs: dict) -> RoutingDecision: ...
