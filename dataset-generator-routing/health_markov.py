from enum import Enum

class HealthState(Enum):
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    OUTAGE = "OUTAGE"

TRANSITIONS = {
    HealthState.HEALTHY: [
        (HealthState.HEALTHY, 0.97),
        (HealthState.DEGRADED, 0.028),
        (HealthState.OUTAGE, 0.002),
    ],

    HealthState.DEGRADED: [
        (HealthState.DEGRADED, 0.85),
        (HealthState.HEALTHY, 0.14),
        (HealthState.OUTAGE, 0.01),
    ],

    HealthState.OUTAGE: [
        (HealthState.OUTAGE, 0.30),
        (HealthState.HEALTHY, 0.70),
    ],
}

import random

def next_health(current_state: HealthState) -> HealthState:

    options = TRANSITIONS[current_state]

    states = [state for state, _ in options]
    probabilities = [prob for _, prob in options]

    return random.choices(
        states,
        weights=probabilities,
        k=1
    )[0]

def generate_health_timeline(minutes: int):

    timeline = []

    current = HealthState.HEALTHY

    for minute in range(minutes):

        timeline.append(current)

        current = next_health(current)

    return timeline

timeline = generate_health_timeline(20)
print(timeline)