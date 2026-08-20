from enum import Enum
import random
from routes import ROUTES


class HealthState(Enum):
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    OUTAGE = "OUTAGE"


def generate_route_timeline(route, minutes: int):

    timeline = {}
    minute = 0

    while minute < minutes:
        if random.random() > route.incident_probability:
            timeline[minute] = HealthState.HEALTHY.value
            minute += 1
            continue

        incident_type = random.choices(
            population=[HealthState.DEGRADED.value, HealthState.OUTAGE.value],
            weights=[route.degraded_probability, 1 - route.degraded_probability],
            k=1,
        )[0]

        duration = random.randint(route.min_incident_duration, route.max_incident_duration)

        for _ in range(duration):
            if minute >= minutes:
                break

            timeline[minute] = incident_type
            minute += 1

    return timeline


def generate_all_health_timelines(minutes: int):
    timelines = {}
    for route in ROUTES:
        timelines[route.route_id] = generate_route_timeline(route, minutes)

    return timelines


if __name__ == "__main__":
    timelines = generate_all_health_timelines(120)
    for route_id, timeline in timelines.items():

        print(f"\n========== {route_id} ==========\n")

        for minute, state in timeline.items():
            print(f"{minute:03d} -> {state.value}")