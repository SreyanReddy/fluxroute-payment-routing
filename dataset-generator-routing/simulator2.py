import random

import numpy as np
import pandas as pd

from routes import ROUTES
from health_incidentbased import HealthState, generate_all_health_timelines
from router import ROUTE_WEIGHTS, choose_route, get_eligible_routes
from transactions import generate_system_tps, generate_transactions


HEALTH_SUCCESS_MULTIPLIER = {
    HealthState.HEALTHY.value: 1.00,
    HealthState.DEGRADED.value: 0.92,
    HealthState.OUTAGE.value: 0.00,
}

HEALTH_CAPACITY_MULTIPLIER = {
    HealthState.HEALTHY.value: 1.00,
    HealthState.DEGRADED.value: 0.60,
    HealthState.OUTAGE.value: 0.00,
}

HEALTH_LATENCY_MULTIPLIER = {
    HealthState.HEALTHY.value: 1.00,
    HealthState.DEGRADED.value: 1.50,
    HealthState.OUTAGE.value: 3.00,
}


def load_success_multiplier(utilization: float) -> float:
    
   ### success-rate penalty caused by overload. Above 70% utilization, success probability is reduced.

    if utilization <= 0.70:
        return 1.00

    if utilization <= 1.00:
        return 1.00 - 0.25 * ((utilization - 0.70) / 0.30)

    return max(0.05, 0.75 - 0.65 * min((utilization - 1.00) / 0.40, 1.00))


def load_latency_multiplier(utilization: float) -> float:

    ### latency growth as utilization exceeds capacity.

    if utilization <= 0.70:
        return 1.00

    if utilization <= 1.00:
        return 1.00 + 2.50 * ((utilization - 0.70) / 0.30) ** 2

    return 3.50 + 6.00 * min(utilization - 1.00, 1.00)


def calculate_route_state(route, health_state: str, assigned_tps: float) -> dict:
    effective_capacity_tps = (route.base_capacity_tps * HEALTH_CAPACITY_MULTIPLIER[health_state])

    if effective_capacity_tps <= 0:
        true_utilization = 1.50
    else:
        true_utilization = assigned_tps / effective_capacity_tps

    # observable utilization -- what a real system can actually measure. 
    # true health is allowed to affect labels, but it should not affect
    # the inputs a production system would have to predict from.

    observed_utilization = assigned_tps / route.base_capacity_tps

    success_probability = (route.base_success_rate * HEALTH_SUCCESS_MULTIPLIER[health_state]
        * load_success_multiplier(true_utilization))

    success_probability = float(np.clip(success_probability, 0.0, 1.0))

    expected_latency_ms = (route.base_latency_ms * HEALTH_LATENCY_MULTIPLIER[health_state]
        * load_latency_multiplier(true_utilization))

    return {
        "health_state": health_state,
        "assigned_tps": assigned_tps,
        "effective_capacity_tps": effective_capacity_tps,
        "utilization": observed_utilization,
        "success_probability": success_probability,
        "expected_latency_ms": expected_latency_ms,
    }


def build_route_metrics(simulation_minutes: int, health_timelines: dict):

    # this simulates production grade route traffic.
    # calculated utilization is available before current minute, no worries
    # outcomes are sampled, so it is safe to expose as a model feature

    route_metrics = []
    states_by_minute = {}

    total_weight = sum(ROUTE_WEIGHTS[route.route_id] for route in ROUTES)

    for minute in range(simulation_minutes):
        system_tps = generate_system_tps(minute)
        minute_states = {}

        for route in ROUTES:
            route_share = (ROUTE_WEIGHTS[route.route_id] / total_weight)

            assigned_tps = system_tps * route_share
            health_state = health_timelines[route.route_id][minute]

            state = calculate_route_state(route, health_state, assigned_tps)

            aggregate_requests = int(round(assigned_tps * 60))

            aggregate_successes = np.random.binomial(aggregate_requests, state["success_probability"])

            aggregate_latency_sum = (aggregate_requests * state["expected_latency_ms"])

            minute_states[route.route_id] = state

            route_metrics.append({
                "minute": minute,
                "route_id": route.route_id,
                "health_state": health_state,

                "aggregate_request_count": aggregate_requests,
                "aggregate_success_count": aggregate_successes,
                "aggregate_latency_sum": aggregate_latency_sum,

                "route_utilization": state["utilization"],
                "route_effective_capacity_tps": (
                    state["effective_capacity_tps"]
                ),
                "system_tps": system_tps,
            })

        states_by_minute[minute] = minute_states

    return states_by_minute, pd.DataFrame(route_metrics)


def sample_latency(expected_latency_ms: float) -> float:
    latency = random.gauss(expected_latency_ms, max(5.0, expected_latency_ms * 0.10))
    return max(1.0, latency)


def transaction_result(transaction, route, route_state: dict) -> dict:

    success = (random.random() < route_state["success_probability"])

    return {
        "txn_id": transaction.txn_id,
        "minute": transaction.minute,
        "amount": transaction.amount,
        "bank": transaction.bank,
        "network": transaction.network,
        "payment_method": transaction.payment_method,
        "merchant": transaction.merchant_category,
        "device": transaction.device,
        "risk": transaction.user_risk,

        "route_id": route.route_id,
        "health_state": route_state["health_state"],

        "success_probability": (
            route_state["success_probability"]
        ),
        "success": int(success),
        "latency_ms": round(
            sample_latency(
                route_state["expected_latency_ms"]
            ),
            2,
        ),
    }


def run_actual_simulation(simulation_minutes: int):

    print("Generating route health...")
    health_timelines = generate_all_health_timelines(simulation_minutes)

    print("Simulating aggregate production traffic...")
    states_by_minute, route_metrics = build_route_metrics(simulation_minutes, health_timelines)

    print("Generating sampled transactions...")
    transactions = generate_transactions(simulation_minutes)
    print(f"Generated {len(transactions):,} sampled rows")

    results = []

    for transaction in transactions:
        route = choose_route(transaction, ROUTES)

        if route is None:
            continue

        route_state = states_by_minute[transaction.minute][route.route_id]

        results.append(transaction_result(transaction, route, route_state))

    return pd.DataFrame(results), route_metrics


def run_counterfactual_simulation(simulation_minutes: int):

    ### four candidate routes for each transaction. used for evaluation.

    print("Generating route health...")
    health_timelines = generate_all_health_timelines(simulation_minutes)

    print("Simulating aggregate production traffic...")
    states_by_minute, route_metrics = build_route_metrics(simulation_minutes, health_timelines)

    print("Generating sampled transactions...")
    transactions = generate_transactions(simulation_minutes)
    print(f"Generated {len(transactions):,} sampled rows")

    results = []

    for transaction in transactions:
        eligible_routes = get_eligible_routes(transaction, ROUTES)

        for route in eligible_routes:
            route_state = states_by_minute[transaction.minute][route.route_id]

            results.append(transaction_result(transaction, route, route_state))

    return pd.DataFrame(results), route_metrics