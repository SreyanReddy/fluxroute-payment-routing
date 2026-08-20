import random
from health_incidentbased import HealthState

HEALTH_SUCCESS_MULTIPLIER = {
    HealthState.HEALTHY: 1.00,
    HealthState.DEGRADED: 0.75,
    HealthState.OUTAGE: 0.10,
}

HEALTH_LATENCY_MULTIPLIER = {
    HealthState.HEALTHY: 1.0,
    HealthState.DEGRADED: 3.0,
    HealthState.OUTAGE: 8.0,
}

RISK_SUCCESS_PENALTY = {
    "LOW": 0.00,
    "MEDIUM": 0.02,
    "HIGH": 0.06,
}

def amount_penalty(amount: float) -> float:
    if amount < 1000:
        return 0.00
    if amount < 5000:
        return 0.01
    if amount < 20000:
        return 0.02
    return 0.04

def calculate_success_probability(
    route,
    transaction,
    health_state,
):
    probability = route.base_success_rate
    probability *= HEALTH_SUCCESS_MULTIPLIER[health_state]
    probability -= RISK_SUCCESS_PENALTY[transaction.user_risk]
    probability -= amount_penalty(transaction.amount)

    probability = max(0.0, min(1.0, probability))
    return probability

def simulate_success(probability: float) -> bool:
    return random.random() < probability

def generate_latency(
    route,
    health_state,
):
    base_latency = route.base_latency_ms
    multiplier = HEALTH_LATENCY_MULTIPLIER[health_state]
    latency = (base_latency * multiplier)
    noise = random.gauss(mu=0, sigma=10)
    latency += noise

    return max(1, round(latency, 2))

def simulate_route_attempt(
    route,
    transaction,
    health_state,
):
    probability = calculate_success_probability(route, transaction, health_state)
    success = simulate_success(probability)
    latency = generate_latency(route, health_state)

    return {
        "route_id": route.route_id,
        "health_state": health_state.value,
        "success_probability": probability,
        "success": int(success),
        "latency_ms": latency,
    }

def simulate_transaction(
    transaction,
    routes,
    health_timelines,
):
    results = []
    minute = transaction.minute
    for route in routes:
        health_state = (
            health_timelines[
                route.route_id
            ][minute]
        )
        result = simulate_route_attempt(route, transaction, health_state)
        results.append(result)
    return results