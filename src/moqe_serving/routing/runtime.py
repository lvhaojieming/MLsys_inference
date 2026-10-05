from typing import Protocol, Sequence


class RouterRuntime(Protocol):
    """Future implementation consumes the complete tokenized prompt.

    Output expert scores must use the checkpoint's expert ordering.
    Scores are not assumed to be calibrated regret estimates.
    """

    def score(self, input_ids: Sequence[int]) -> dict[str, float]: ...
