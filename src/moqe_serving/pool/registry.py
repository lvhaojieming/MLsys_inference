from ..config import Replica
from .models import InstanceState, ALLOWED_TRANSITIONS
import time
import logging


class Registry:
    """Dynamic membership; only ready replicas may acquire a lease."""

    def __init__(self, replicas: tuple[Replica, ...], require_admission=True):
        if replicas and not require_admission:
            raise ValueError("Instances must start in STARTING and pass admission")
        self.replicas = replicas
        self.inflight = {r.id: 0 for r in replicas}
        self._cursor = 0
        self.states = {r.id: InstanceState.STARTING for r in replicas}
        self.validation = {}
        self.validating = set()
        self.transitions = []

    def transition(self, replica_id, target, reason):
        target = InstanceState(target)
        source = self.states[replica_id]
        if source == target:
            return
        if target not in ALLOWED_TRANSITIONS[source]:
            raise ValueError(f"Illegal instance transition: {source.value} -> {target.value}")
        if target == InstanceState.OFFLINE and self.inflight[replica_id]:
            raise ValueError("Cannot mark an instance offline with outstanding requests")
        self.states[replica_id] = target
        event = {"id": replica_id, "from": source.value, "to": target.value,
                 "reason": reason, "timestamp": time.time()}
        self.transitions.append(event)
        logging.getLogger(__name__).info("instance_transition %s", event)

    def begin_validation(self, replica_id):
        state = self.states[replica_id]
        if replica_id in self.validating or state not in {
                InstanceState.STARTING, InstanceState.UNHEALTHY, InstanceState.OFFLINE}:
            raise ValueError(f"Cannot validate instance in state {state.value}")
        if state == InstanceState.OFFLINE:
            self.transition(replica_id, InstanceState.STARTING, "restart requested")
        self.validating.add(replica_id)

    def finish_validation(self, replica_id, passed):
        self.validating.discard(replica_id)
        # Drain wins over a concurrent admission or recovery result.
        if self.states[replica_id] in {InstanceState.STARTING, InstanceState.UNHEALTHY}:
            self.transition(replica_id, InstanceState.READY if passed else InstanceState.UNHEALTHY,
                            "admission/recovery passed" if passed else "admission/recovery failed")

    def replace_offline(self, replica):
        if self.states[replica.id] != InstanceState.OFFLINE or replica.id in self.validating:
            raise ValueError("Replacement requires an offline instance with completed validation")
        self.replicas = tuple(replica if r.id == replica.id else r for r in self.replicas)
        self.validation.pop(replica.id, None)
        self.transition(replica.id, InstanceState.STARTING, "configuration re-enabled/replaced instance")

    def ready_instances(self, expert):
        return [r for r in self.replicas if r.expert == expert and self.states[r.id] == InstanceState.READY]

    def add(self, replica):
        if replica.id in self.states:
            raise ValueError("Replica id already registered")
        self.replicas = (*self.replicas, replica)
        self.inflight[replica.id] = 0
        self.states[replica.id] = InstanceState.STARTING

    def get(self, replica_id):
        return next(r for r in self.replicas if r.id == replica_id)

    def drain(self, replica_id):
        if replica_id not in self.states:
            raise KeyError(replica_id)
        if self.states[replica_id] == InstanceState.OFFLINE:
            return
        self.transition(replica_id, InstanceState.DRAINING, "drain requested; stop new leases")
        if not self.inflight[replica_id]:
            self.transition(replica_id, InstanceState.OFFLINE, "all leases released")

    def snapshot(self):
        return [{"id": r.id, "expert": r.expert, "model": r.model,
                 "node_id": r.node_id, "device_ids": r.device_ids,
                 "base_url": r.base_url, "state": self.states[r.id],
                 "inflight": self.inflight[r.id], "validation_in_progress": r.id in self.validating,
                 "validation": self.validation.get(r.id)}
                for r in self.replicas]

    @property
    def experts(self):
        return sorted({r.expert for r in self.replicas})

    def acquire(self, expert: str):
        candidates = self.ready_instances(expert)
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
        if self.states[replica.id] == "draining" and self.inflight[replica.id] == 0:
            self.transition(replica.id, InstanceState.OFFLINE, "last outstanding request completed")
