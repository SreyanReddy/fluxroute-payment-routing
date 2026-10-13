import random

from locust import HttpUser, between, task

BANKS = ["HDFC", "ICICI", "SBI", "Axis", "Kotak", "BOB"]
BANK_WEIGHTS = [30, 18, 22, 12, 10, 8]

CARD_NETWORKS = ["Visa", "Mastercard", "Rupay"]
PAYMENT_METHODS = ["CARD", "UPI"]

MERCHANTS = ["Food", "Shopping", "Travel", "Electronics", "Fuel", "Healthcare"]
MERCHANT_WEIGHTS = [25, 30, 8, 10, 15, 12]

DEVICES = ["Android", "iOS", "Web"]
RISK_LEVELS = ["LOW", "MEDIUM", "HIGH"]

ROUTE_BASE_SUCCESS_RATE = {
    "R1": 0.986,
    "R2": 0.981,
    "R3": 0.979,
    "R4": 0.983,
}

ROUTE_BASE_LATENCY_MS = {
    "R1": 145,
    "R2": 85,
    "R3": 110,
    "R4": 175,
}


def generate_risk(amount: float) -> str:
    if amount < 1000:
        weights = [85, 13, 2]
    elif amount < 5000:
        weights = [60, 30, 10]
    else:
        weights = [30, 45, 25]
    return random.choices(RISK_LEVELS, weights=weights, k=1)[0]


def generate_transaction() -> dict:
    amount = round(random.lognormvariate(5.5, 1.35), 2)
    amount = max(10.0, min(amount, 50000.0))

    payment_method = random.choice(PAYMENT_METHODS)
    network = random.choice(CARD_NETWORKS) if payment_method == "CARD" else "UPI"

    return {
        "amount": amount,
        "bank": random.choices(BANKS, weights=BANK_WEIGHTS, k=1)[0],
        "network": network,
        "payment_method": payment_method,
        "merchant": random.choices(MERCHANTS, weights=MERCHANT_WEIGHTS, k=1)[0],
        "device": random.choice(DEVICES),
        "risk": generate_risk(amount),
    }


def simulate_outcome(route_id: str) -> tuple[bool, float]:
    success_rate = ROUTE_BASE_SUCCESS_RATE.get(route_id, 0.98)
    base_latency = ROUTE_BASE_LATENCY_MS.get(route_id, 120)
    success = random.random() < success_rate
    latency_ms = max(1.0, random.gauss(base_latency, base_latency * 0.15))
    return success, round(latency_ms, 2)


class RoutingUser(HttpUser):
    wait_time = between(0.05, 0.2)

    @task
    def route_and_report(self):
        payload = generate_transaction()

        with self.client.post(
            "/route", json=payload, name="POST /route", catch_response=True
        ) as response:
            if response.status_code != 200:
                response.failure(f"status={response.status_code}")
                return

            body = response.json()
            transaction_id = body["transaction_id"]
            chosen_route_id = body["chosen_route_id"]
            response.success()

        success, latency_ms = simulate_outcome(chosen_route_id)

        self.client.post(
            "/outcome",
            json={
                "transaction_id": transaction_id,
                "success": success,
                "latency_ms": latency_ms,
            },
            name="POST /outcome",
        )
