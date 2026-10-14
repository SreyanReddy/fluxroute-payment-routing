import json
import logging
import os
import sys
import time
import uuid
import zlib

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
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

# Fraction of transactions whose route/outcome events are logged (errors are
# always logged). 1.0 logs everything; load tests use a small fraction so that
# writing logs does not become part of what is being measured.
LOG_SAMPLE_RATE = float(os.environ.get("LOG_SAMPLE_RATE", "1.0"))

app = FastAPI(title="FluxRoute Smart Payment Routing")

_artifact = joblib.load(MODEL_PATH)
_model = _artifact["model"]
_preprocessor = _artifact["preprocessor"]
_calibrator = joblib.load(CALIBRATOR_PATH)
_feature_columns = _artifact["numeric_features"] + _artifact["categorical_features"]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

# Request counters. status is "ok" or the reason the request failed, so the
# error rate is (status != "ok") / all.
#   /route:   ok | no_eligible_route | invalid_request | internal_error
#   /outcome: ok | unknown_transaction | invalid_request | internal_error
ROUTE_REQUESTS = Counter(
    "fluxroute_route_requests_total", "Routing requests by result", ["instance", "status"]
)
OUTCOME_REQUESTS = Counter(
    "fluxroute_outcome_requests_total", "Outcome reports by result", ["instance", "status"]
)

# Latency histograms. Buckets are packed closely around the range measured in
# the baseline sweep, so percentile estimates do not jump across wide buckets.
ROUTE_LATENCY = Histogram(
    "fluxroute_route_latency_seconds",
    "End-to-end latency of POST /route",
    ["instance"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0, 2.5),
)
SCORING_LATENCY = Histogram(
    "fluxroute_scoring_latency_seconds",
    "Model scoring time excluding feature fetch",
    ["instance"],
    buckets=(0.0005, 0.001, 0.0025, 0.005, 0.01, 0.015, 0.02, 0.025, 0.05, 0.1, 0.25),
)
FEATURE_FETCH_LATENCY = Histogram(
    "fluxroute_feature_fetch_seconds",
    "Redis feature retrieval time per request",
    ["instance"],
    buckets=(0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0),
)

# Routing behaviour.
ROUTE_SELECTED = Counter(
    "fluxroute_route_selected_total", "Routing decisions by route", ["instance", "route_id"]
)
CHOSEN_PROBABILITY = Histogram(
    "fluxroute_chosen_success_probability",
    "Calibrated success probability of the chosen route (watch for distribution drift)",
    ["instance"],
    buckets=(0.5, 0.8, 0.9, 0.95, 0.97, 0.98, 0.985, 0.99, 0.995, 1.0),
)

# Outcomes and calibration. For the resolved transactions of a route:
#   observed success rate = successes / outcomes
#   expected success rate = sum of predicted probabilities / outcomes
# If the model is well calibrated on live traffic, the two lines overlap.
OUTCOMES = Counter(
    "fluxroute_outcomes_total", "Recorded outcomes", ["instance", "route_id", "result"]
)
PREDICTED_SUCCESS = Counter(
    "fluxroute_predicted_success_total",
    "Sum of the predicted success probabilities of resolved transactions",
    ["instance", "route_id"],
)

# Traffic meter.
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

_REQUEST_COUNTERS = {"/route": ROUTE_REQUESTS, "/outcome": OUTCOME_REQUESTS}


def count_request(path: str, status: str) -> None:
    counter = _REQUEST_COUNTERS.get(path)
    if counter is not None:
        counter.labels(instance=INSTANCE_ID, status=status).inc()


# ---------------------------------------------------------------------------
# Structured logging: one JSON object per line on stdout
# ---------------------------------------------------------------------------
#
# Read them with:  docker compose logs app1 --no-log-prefix | jq

_logger = logging.getLogger("fluxroute")
_log_handler = logging.StreamHandler(sys.stdout)
_log_handler.setFormatter(logging.Formatter("%(message)s"))
_logger.addHandler(_log_handler)
_logger.setLevel(logging.INFO)
_logger.propagate = False


def log_event(level: int, event: str, **fields) -> None:
    record = {
        "ts": round(time.time(), 3),
        "level": logging.getLevelName(level).lower(),
        "event": event,
        "instance": INSTANCE_ID,
        **fields,
    }
    _logger.log(level, json.dumps(record, default=str))


def is_sampled(transaction_id: str) -> bool:
    # Decided from the transaction id itself, so /route and /outcome make the
    # same choice even when they land on different replicas. (Python's hash()
    # would not work: it is randomised per process.)
    return zlib.crc32(transaction_id.encode()) % 10_000 < LOG_SAMPLE_RATE * 10_000


# ---------------------------------------------------------------------------
# Error handlers: count and log every failed request, including crashes
# ---------------------------------------------------------------------------


@app.exception_handler(RequestValidationError)
async def on_invalid_request(request: Request, exc: RequestValidationError):
    count_request(request.url.path, "invalid_request")
    # Log which fields were wrong, never the submitted values: in a payments
    # service the request body can carry data that must not reach logs.
    errors = [{"field": ".".join(map(str, e["loc"])), "problem": e["msg"]} for e in exc.errors()]
    log_event(logging.WARNING, "invalid_request", path=request.url.path, errors=errors)
    return await request_validation_exception_handler(request, exc)


@app.exception_handler(Exception)
async def on_unhandled_error(request: Request, exc: Exception):
    count_request(request.url.path, "internal_error")
    log_event(
        logging.ERROR,
        "internal_error",
        path=request.url.path,
        error_type=type(exc).__name__,
        error=str(exc),
    )
    return JSONResponse(status_code=500, content={"detail": "internal error"})


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------


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
        count_request("/route", "no_eligible_route")
        log_event(
            logging.WARNING,
            "no_eligible_route",
            transaction_id=transaction_id,
            payment_method=request.payment_method,
            network=request.network,
        )
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
    fetch_seconds = time.perf_counter() - fetch_started
    FEATURE_FETCH_LATENCY.labels(instance=INSTANCE_ID).observe(fetch_seconds)

    candidates_df = pd.DataFrame(rows)

    scoring_started = time.perf_counter()
    X = candidates_df[_feature_columns]
    raw_probabilities = _model.predict_proba(_preprocessor.transform(X))[:, 1]
    scoring_seconds = time.perf_counter() - scoring_started
    SCORING_LATENCY.labels(instance=INSTANCE_ID).observe(scoring_seconds)

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
    record_pending_route(
        transaction_id, chosen_route_id, request.bank, request.network, chosen_probability
    )

    total_seconds = time.perf_counter() - started
    ROUTE_SELECTED.labels(instance=INSTANCE_ID, route_id=chosen_route_id).inc()
    CHOSEN_PROBABILITY.labels(instance=INSTANCE_ID).observe(chosen_probability)
    count_request("/route", "ok")
    ROUTE_LATENCY.labels(instance=INSTANCE_ID).observe(total_seconds)

    if is_sampled(transaction_id):
        log_event(
            logging.INFO,
            "route_decision",
            transaction_id=transaction_id,
            bank=request.bank,
            network=request.network,
            payment_method=request.payment_method,
            amount=request.amount,
            chosen_route_id=chosen_route_id,
            predicted_success_probability=round(chosen_probability, 4),
            candidates={
                str(route_id): round(float(probability), 4)
                for route_id, probability in zip(
                    candidates_df["route_id"], candidates_df["predicted_success_probability"]
                )
            },
            system_tps=round(system_tps),
            system_tps_clamp=clamp,
            fetch_ms=round(fetch_seconds * 1000, 2),
            scoring_ms=round(scoring_seconds * 1000, 2),
            total_ms=round(total_seconds * 1000, 2),
        )

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
        # Unknown id, an outcome reported twice, or one that arrived after the
        # pending decision expired.
        count_request("/outcome", "unknown_transaction")
        log_event(logging.WARNING, "unknown_transaction", transaction_id=request.transaction_id)
        raise HTTPException(
            status_code=404,
            detail=(
                "No pending routing decision found for "
                f"transaction_id={request.transaction_id}"
            ),
        )

    route_id = pending["route_id"]
    record_transaction_outcome(
        route_id=route_id,
        bank=pending["bank"],
        network=pending["network"],
        success=request.success,
        latency_ms=request.latency_ms,
    )

    OUTCOMES.labels(
        instance=INSTANCE_ID,
        route_id=route_id,
        result="success" if request.success else "failure",
    ).inc()

    # Decisions stored before this field existed have no probability; they
    # are left out of the calibration comparison rather than guessed.
    stored_probability = pending.get("predicted_probability")
    predicted_probability = float(stored_probability) if stored_probability is not None else None
    if predicted_probability is not None:
        PREDICTED_SUCCESS.labels(instance=INSTANCE_ID, route_id=route_id).inc(
            predicted_probability
        )

    count_request("/outcome", "ok")

    if is_sampled(request.transaction_id):
        log_event(
            logging.INFO,
            "outcome",
            transaction_id=request.transaction_id,
            route_id=route_id,
            success=request.success,
            latency_ms=request.latency_ms,
            predicted_success_probability=predicted_probability,
        )

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
