from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class Route:

    route_id: str
    name: str

    # Baseline characteristics
    base_success_rate: float      
    base_latency_ms: int          

    # Business characteristic
    cost_percent: float          

    # Compatibility
    supported_payment_methods: List[str]
    supported_networks: List[str]

    incident_probability: float

    degraded_probability: float

    min_incident_duration: int

    max_incident_duration: int

    #V2
    base_capacity_tps: int

ROUTES = [
    Route(
        route_id="R1",
        name="Gateway Alpha",

        base_success_rate=0.986,
        base_latency_ms=145,
        cost_percent=2.10,

        supported_payment_methods=["CARD", "UPI"],
        supported_networks=["Visa", "Mastercard", "UPI"],

        incident_probability=0.006,
        degraded_probability=0.90,
        min_incident_duration=5,
        max_incident_duration=15,
        base_capacity_tps=120_000,
    ),

    Route(
        route_id="R2",
        name="Gateway Beta",

        base_success_rate=0.981,
        base_latency_ms=85,
        cost_percent=1.45,

        supported_payment_methods=["CARD", "UPI"],
        supported_networks=["Visa", "Rupay", "UPI"],

        incident_probability=0.010,
        degraded_probability=0.85,
        min_incident_duration=5,
        max_incident_duration=20,
        base_capacity_tps=150_000,
    ),

    Route(
        route_id="R3",
        name="Gateway Gamma",

        base_success_rate=0.979,
        base_latency_ms=110,
        cost_percent=1.10,

        supported_payment_methods=["CARD", "UPI"],
        supported_networks=["Visa", "Mastercard","Rupay", "UPI"],

        incident_probability=0.008,
        degraded_probability=0.92,
        min_incident_duration=8,
        max_incident_duration=18,
        base_capacity_tps=100_000,
    ),

    Route(
        route_id="R4",
        name="Gateway Delta",

        base_success_rate=0.983,
        base_latency_ms=175,
        cost_percent=1.75,

        supported_payment_methods=["CARD", "UPI"],
        supported_networks=["Visa", "Mastercard", "Rupay", "UPI"],

        incident_probability=0.007,
        degraded_probability=0.75,
        min_incident_duration=10,
        max_incident_duration=30,
        base_capacity_tps=75_000,
    ),
]