from ..config import Replica


class Registry:
    """Static registry; lease counts apply to one gateway process."""

    def __init__(self, replicas: tuple[Replica, ...]):
        self.replicas = replicas
        self.inflight = {r.id: 0 for r in replicas}
        self._cursor = 0

    @property
    def experts(self):
        return sorted({r.expert for r in self.replicas})

    def acquire(self, expert: str):
        candidates = [r for r in self.replicas if r.expert == expert]
        if not candidates:
            raise KeyError(expert)
        minimum = min(self.inflight[r.id] for r in candidates)
        tied = [r for r in candidates if self.inflight[r.id] == minimum]
        replica = tied[self._cursor % len(tied)]
        self._cursor += 1
        self.inflight[replica.id] += 1
        return replica

    def release(self, replica: Replica):
        if self.inflight[replica.id] <= 0:
            raise RuntimeError("Replica lease already released")
        self.inflight[replica.id] -= 1
