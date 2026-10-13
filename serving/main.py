import os
import sys
import time
import uuid

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)

from feature_storing import (  # noqa: E402
    current_second,
    read_system_tps,
    record_ingress,
    get_eligible_routes,
    build_route_features,
    record_routing_decision,
    record_transaction_outcome,
    record_pending_route,
    pop_pending_route,
    get_redis_client,
)

MODEL_PATH = os.environ.get(
    "MODEL_PATH", os.path.join(THIS_DIR, "..", "models", "lightgbm_tuned_v2.joblib")
)
CALIBRATOR_PATH = os.environ.get(
    "CALIBRATOR_PATH",
    os.path.join(THIS_DIR, "..", "models", "isotonic_calibrator_v2.joblib"),
)
INSTANCE_ID = os.environ.get("INSTANCE_ID", "local")

app = FastAPI(title="FluxRoute Smart Payment Routing")

_artifact = joblib.load(MODEL_PATH)
_model = _artifact["model"]
_preprocessor = _artifact["preprocessor"]
_calibrator = joblib.load(CALIBRATOR_PATH)
_feature_columns = _artifact["numeric_features"] + _artifact["categorical_features"]

ROUTE_REQUESTS = Counter(
    "fluxroute_route_requests_total", "Routing requests served", ["instance", "status"]
)
ROUTE_LATENCY = Histogram(
    "fluxroute_route_latency_seconds",
    "End-to-end latency of POST /route",
    ["instance"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 1.0, 2.5),
)
SCORING_LATENCY = Histogram(
    "fluxroute_scoring_latency_seconds",
    "Model scoring time excluding feature fetch",
    ["instance"],
    buckets=(0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25),
)
FEATURE_FETCH_LATENCY = Histogram(
    "fluxroute_feature_fetch_seconds",
    "Redis feature retrieval time per request",
    ["instance"],
    buckets=(0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25),
)
ROUTE_SELECTED = Counter(
    "fluxroute_route_selected_total", "Routing decisions by route", ["instance", "route_id"]
)
OUTCOMES = Counter(
    "fluxroute_outcomes_total", "Recorded outcomes", ["instance", "route_id", "result"]
)
SYSTEM_TPS = Gauge(
    "fluxroute_system_tps",
    "Traffic-meter reading passed to the model (scaled and clamped)",
    ["instance"],
)
SYSTEM_TPS_CLAMPED = Counter(
    "fluxroute_system_tps_clamped_total",
    "Requests whose traffic reading hit the training range limit; "
    "frequent clamping means TRAFFIC_SCALE is wrong for the load being run",
    ["instance", "direction"],
)


class RouteRequest(BaseModel):
    transaction_id: str | None = None
    amount: float
    bank: str
    network: str
    payment_method: str
    merchant: str
    device: str
    risk: str


class CandidateScore(BaseModel):
    route_id: str
    predicted_success_probability: float


class RouteResponse(BaseModel):
    transaction_id: str
    chosen_route_id: str
    predicted_success_probability: float
    candidates: list[CandidateScore]


class OutcomeRequest(BaseModel):
    transaction_id: str
    success: bool
    latency_ms: float


@app.post("/route", response_model=RouteResponse)
def route_transaction(request: RouteRequest) -> RouteResponse:
    started = time.perf_counter()

    transaction_id = request.transaction_id or str(uuid.uuid4())
    eligible_routes = get_eligible_routes(request.payment_method, request.network)

    if not eligible_routes:
        ROUTE_REQUESTS.labels(instance=INSTANCE_ID, status="no_eligible_route").inc()
        raise HTTPException(
            status_code=422,
            detail=(
                f"No route supports payment_method={request.payment_method} "
                f"network={request.network}"
            ),
        )

    fetch_started = time.perf_counter()

    now_second = current_second()
    system_tps, clamp = read_system_tps(now_second)
    record_ingress(now_second)
    SYSTEM_TPS.labels(instance=INSTANCE_ID).set(system_tps)
    if clamp is not None:
        SYSTEM_TPS_CLAMPED.labels(instance=INSTANCE_ID, direction=clamp).inc()

    rows = [
        build_route_features(
            route_id=route_id,
            bank=request.bank,
            network=request.network,
            amount=request.amount,
            payment_method=request.payment_method,
            merchant=request.merchant,
            device=request.device,
            risk=request.risk,
            system_tps=system_tps,
        )
        for route_id in eligible_routes
    ]
    FEATURE_FETCH_LATENCY.labels(instance=INSTANCE_ID).observe(
        time.perf_counter() - fetch_started
    )

    candidates_df = pd.DataFrame(rows)

    scoring_started = time.perf_counter()
    X = candidates_df[_feature_columns]
    raw_probabilities = _model.predict_proba(_preprocessor.transform(X))[:, 1]
    SCORING_LATENCY.labels(instance=INSTANCE_ID).observe(
        time.perf_counter() - scoring_started
    )

    candidates_df["raw_success_probability"] = raw_probabilities
    candidates_df["predicted_success_probability"] = _calibrator.predict(raw_probabilities)

    best_index = candidates_df["raw_success_probability"].idxmax()
    chosen_route_id = str(candidates_df.loc[best_index, "route_id"])
    chosen_probability = float(
        candidates_df.loc[best_index, "predicted_success_probability"] # type: ignore
    )

    record_routing_decision(
        route_id=chosen_route_id,
        bank=request.bank,
        network=request.network,
        amount=request.amount,
    )
    record_pending_route(transaction_id, chosen_route_id, request.bank, request.network)

    ROUTE_SELECTED.labels(instance=INSTANCE_ID, route_id=chosen_route_id).inc()
    ROUTE_REQUESTS.labels(instance=INSTANCE_ID, status="ok").inc()
    ROUTE_LATENCY.labels(instance=INSTANCE_ID).observe(time.perf_counter() - started)

    return RouteResponse(
        transaction_id=transaction_id,
        chosen_route_id=chosen_route_id,
        predicted_success_probability=chosen_probability,
        candidates=[
            CandidateScore(
                route_id=str(row["route_id"]),
                predicted_success_probability=float(row["predicted_success_probability"]),
            )
            for _, row in candidates_df.iterrows()
        ],
    )


@app.post("/outcome")
def record_outcome(request: OutcomeRequest) -> dict:
    pending = pop_pending_route(request.transaction_id)

    if pending is None:
        raise HTTPException(
            status_code=404,
            detail=(
                "No pending routing decision found for "
                f"transaction_id={request.transaction_id}"
            ),
        )

    record_transaction_outcome(
        route_id=pending["route_id"],
        bank=pending["bank"],
        network=pending["network"],
        success=request.success,
        latency_ms=request.latency_ms,
    )

    OUTCOMES.labels(
        instance=INSTANCE_ID,
        route_id=pending["route_id"],
        result="success" if request.success else "failure",
    ).inc()

    return {"status": "recorded", "transaction_id": request.transaction_id}


@app.get("/health")
def health_check() -> dict:
    try:
        get_redis_client().ping()
        redis_ok = True
    except Exception:
        redis_ok = False

    return {
        "status": "ok" if redis_ok else "degraded",
        "redis_connected": redis_ok,
        "instance": INSTANCE_ID,
    }


@app.get("/metrics")
def metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
