from dataclasses import dataclass
import numpy as np
import random


@dataclass
class Transaction:

    txn_id: int
    minute: int
    amount: float
    bank: str
    network: str
    payment_method: str
    merchant_category: str
    device: str
    user_risk: str

TRAFFIC_MULTIPLIER = {
    0: 0.30,
    1: 0.20,
    2: 0.10,
    3: 0.10,
    4: 0.15,
    5: 0.25,

    6: 0.45,
    7: 0.65,
    8: 0.85,
    9: 1.10,
    10: 1.40,
    11: 1.70,

    12: 1.80,
    13: 1.50,
    14: 1.10,
    15: 0.95,
    16: 1.10,

    17: 1.40,
    18: 1.75,
    19: 2.00,
    20: 2.10,
    21: 1.80,
    22: 1.10,
    23: 0.50,
}

BANKS = [
    "HDFC",
    "ICICI",
    "SBI",
    "Axis",
    "Kotak",
    "BOB",
]

BANK_WEIGHTS = [
    30,
    18,
    22,
    12,
    10,
    8,
]

NETWORKS = [
    "Visa",
    "Mastercard",
    "Rupay",
]

PAYMENT_METHODS = [
    "CARD",
    "UPI",
]

MERCHANTS = [
    "Food",
    "Shopping",
    "Travel",
    "Electronics",
    "Fuel",
    "Healthcare",
]

MERCHANT_WEIGHTS = [
    25,
    30,
    8,
    10,
    15,
    12,
]

DEVICES = [
    "Android",
    "iOS",
    "Web",
]

RISK_LEVELS = [
    "LOW",
    "MEDIUM",
    "HIGH",
]

def generate_risk(amount):
    if amount < 1000:
        weights = [85, 13, 2]

    elif amount < 5000:
        weights = [60, 30, 10]

    else:
        weights = [30, 45, 25]

    return random.choices(RISK_LEVELS, weights=weights, k=1)[0]

BASE_SAMPLED_TRANSACTIONS_PER_MINUTE = 15
BASE_SYSTEM_TPS = 100_000
WEEKEND_MULTIPLIER = 1.28

def generate_transaction(txn_id, minute):
    amount = round(random.lognormvariate(5.5, 1.35), 2)
    amount = max(10, min(amount, 50000))
    payment_method=random.choice(PAYMENT_METHODS)
    if payment_method == "CARD":
        network = random.choice(["Visa", "Mastercard", "Rupay"])
    else:
        network = "UPI"
    
    return Transaction(
        txn_id=txn_id,
        minute=minute,
        amount=amount,
        bank=random.choices(
            BANKS,
            weights=BANK_WEIGHTS,
            k=1,
        )[0],
        payment_method=payment_method, 
        network=network,
        merchant_category=random.choices(
            MERCHANTS,
            weights=MERCHANT_WEIGHTS,
            k=1,
        )[0],

        device=random.choice(DEVICES),
        user_risk=generate_risk(amount),
    )

def get_traffic_multiplier(minute: int) -> float:
    hour = (minute // 60) % 24
    day_of_week = (minute // (24 * 60)) % 7

    multiplier = TRAFFIC_MULTIPLIER[hour]

    if day_of_week >= 5:
        multiplier *= WEEKEND_MULTIPLIER

    return multiplier


def generate_system_tps(minute: int) -> float:
    expected_tps = BASE_SYSTEM_TPS * get_traffic_multiplier(minute)

    # Small traffic variation around the expected rate.
    return max(1.0, np.random.normal(loc=expected_tps, scale=expected_tps * 0.05))

def generate_transaction_count(minute: int) -> int:
    expected_samples = (BASE_SAMPLED_TRANSACTIONS_PER_MINUTE * get_traffic_multiplier(minute))

    return np.random.poisson(expected_samples)

def generate_transactions(minutes):
    transactions = []
    txn_id = 1
    for minute in range(minutes):
        txns_this_minute = generate_transaction_count(minute)
        for _ in range(txns_this_minute):
            transactions.append(generate_transaction(txn_id, minute))
            txn_id += 1
    return transactions