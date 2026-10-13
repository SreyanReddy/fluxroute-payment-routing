"""
Writes models/serving_constants_v2.json: every per-route number the serving
code needs, taken from the same sources training used.

Why this exists:
    The serving code used to type these numbers in by hand. It typed in the
    simulator's nominal settings (R4 latency 175 ms) instead of what training
    actually measured (R4 latency 283.69 ms). The model learned from the
    measured values, so serving fed it wrong inputs. Generating the file from
    the training data makes that mistake impossible.

Where each number comes from:
    base_success_rate, base_latency_ms, cost_percent
        the training rows of the feature CSV (exactly what the model saw)
    base_capacity_tps, supported_payment_methods, supported_networks
        dataset-generator-routing/routes.py
    routing_share
        dataset-generator-routing/router.py (ROUTE_WEIGHTS)
    system_tps_training
        min / median / max of system_tps over the training rows

Run:
    ~/.venvs/fluxroute/bin/python ml/export_serving_constants.py
"""

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GENERATOR_DIR = PROJECT_ROOT / "dataset-generator-routing"
sys.path.insert(0, str(GENERATOR_DIR))

from router import ROUTE_WEIGHTS  # noqa: E402
from routes import ROUTES  # noqa: E402

FEATURES_CSV = GENERATOR_DIR / "payment_dataset_features_4week_v2.csv"
OUTPUT = PROJECT_ROOT / "models" / "serving_constants_v2.json"


def main() -> None:
    rows = pd.read_csv(
        FEATURES_CSV,
        usecols=[
            "route_id",
            "route_base_success_rate",
            "route_base_latency_ms",
            "route_cost_percent",
            "system_tps",
            "dataset_split",
        ],
    )
    training = rows[rows["dataset_split"] == "train"]
    measured = training.groupby("route_id")[
        ["route_base_success_rate", "route_base_latency_ms", "route_cost_percent"]
    ].first()

    routes = {}
    for route in ROUTES:
        r = route.route_id
        routes[r] = {
            "base_success_rate": float(measured.loc[r, "route_base_success_rate"]),
            "base_latency_ms": float(measured.loc[r, "route_base_latency_ms"]),
            "cost_percent": float(measured.loc[r, "route_cost_percent"]),
            "base_capacity_tps": int(route.base_capacity_tps),
            "routing_share": float(ROUTE_WEIGHTS[r]),
            "supported_payment_methods": sorted(route.supported_payment_methods),
            "supported_networks": sorted(route.supported_networks),
        }

    constants = {
        "model": "models/lightgbm_tuned_v2.joblib",
        "source": "training rows of dataset-generator-routing/payment_dataset_features_4week_v2.csv",
        "routes": routes,
        "system_tps_training": {
            "min": float(training["system_tps"].min()),
            "median": float(training["system_tps"].median()),
            "max": float(training["system_tps"].max()),
        },
    }

    OUTPUT.write_text(json.dumps(constants, indent=2) + "\n")
    print(f"Wrote {OUTPUT.relative_to(PROJECT_ROOT)}")
    print(json.dumps(constants, indent=2))


if __name__ == "__main__":
    main()
