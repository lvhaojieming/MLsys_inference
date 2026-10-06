from enum import Enum


class InstanceState(str, Enum):
    STARTING = "starting"
    READY = "ready"
    DRAINING = "draining"
    UNHEALTHY = "unhealthy"
    OFFLINE = "offline"


# Normal lifecycle is exactly the report's diagram. Extra edges handle startup
# failure, cancellation before readiness, and restarting a previously offline replica.
ALLOWED_TRANSITIONS = {
    InstanceState.STARTING: {InstanceState.READY, InstanceState.UNHEALTHY, InstanceState.DRAINING},
    InstanceState.READY: {InstanceState.DRAINING, InstanceState.UNHEALTHY},
    InstanceState.DRAINING: {InstanceState.OFFLINE},
    InstanceState.UNHEALTHY: {InstanceState.READY, InstanceState.DRAINING},
    InstanceState.OFFLINE: {InstanceState.STARTING},
}
