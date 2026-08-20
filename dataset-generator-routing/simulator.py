import pandas as pd
import random

from routes import ROUTES
from health_incidentbased import generate_all_health_timelines, HealthState
from transactions import generate_transactions
from router import choose_route

def run_counterfactual_simulation(simulation_minutes: int):
    print("Generating route health...")

    health_timelines = generate_all_health_timelines(simulation_minutes)

    print("Generating transactions...")
    transactions = generate_transactions(simulation_minutes)
    print(f"Generated {len(transactions)} transactions")

    results = []
    for transaction in transactions:
        route_results = simulate_counterfactual_transaction(transaction, ROUTES, health_timelines)
        results.extend(route_results)

    return pd.DataFrame(results)


def simulate_counterfactual_transaction(transaction, routes, health_timelines):

    results = []
    for route in routes:
        health_state = health_timelines[route.route_id][transaction.minute]
        success_probability = calculate_success_probability(route, health_state)
        success = (random.random() < success_probability)

        latency = simulate_latency(route, health_state)

        results.append({
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
            "health_state": health_state,

            "success_probability": success_probability,
            "success": int(success),
            "latency_ms": round(latency, 2)
        })

    return results

def run_actual_simulation(simulation_minutes: int):

    print("Generating route health...")
    health_timelines = generate_all_health_timelines(simulation_minutes)

    print("Generating transactions...")
    transactions = generate_transactions(simulation_minutes)
    print(f"Generated {len(transactions)} transactions")

    results = []
    for transaction in transactions:
        result = simulate_actual_transaction(transaction, ROUTES, health_timelines)
        if result is not None:
            results.append(result)

    return pd.DataFrame(results)


def simulate_actual_transaction(transaction, routes, health_timelines):

    route = choose_route(transaction, routes)

    if route is None:
        return None

    health_state = health_timelines[route.route_id][transaction.minute]
    success_probability = calculate_success_probability(route,health_state)
    success = (random.random() < success_probability)
    latency = simulate_latency(route, health_state)

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
        "health_state": health_state,
        "success_probability": success_probability,
        "success": int(success),
        "latency_ms": round(latency, 2)
    }

def calculate_success_probability(route, health_state):

    if health_state == HealthState.HEALTHY.value:
        return route.base_success_rate
    elif health_state == HealthState.DEGRADED.value:
        return route.base_success_rate * 0.80
    else:
        return 0.0


def simulate_latency(route, health_state):

    if health_state == HealthState.HEALTHY.value:
        latency = random.gauss(route.base_latency_ms, route.base_latency_ms * 0.10)
    elif health_state == HealthState.DEGRADED.value:
        latency = random.gauss(route.base_latency_ms * 2, route.base_latency_ms * 0.20)
    else:
        latency = random.gauss(route.base_latency_ms * 3, route.base_latency_ms * 0.30)

    return max(1, latency)

