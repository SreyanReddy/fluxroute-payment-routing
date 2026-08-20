"""
FastAPI routing service.

POST /route    -- score all eligible candidate routes for a transaction,
                   return the argmax choice.
POST /outcome  -- record the real outcome once known, updating the Redis
                   state feature_store.py's rolling features read from.
"""

import os
import sys
import uuid

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)
sys.path.insert(0, os.path.join(THIS_DIR, "..", "ml"))

from lightgbm_tuned import NUMERIC_FEATURES, CATEGORICAL_FEATURES  # noqa: E402
from feature_storing import (  # noqa: E402
    get_eligible_routes,
    build_route_features,
    record_routing_decision,
    record_transaction_outcome,
    record_pending_route,
    pop_pending_route,
    get_redis_client,
)

MODEL_PATH = os.path.join(THIS_DIR, "..", "models", "lightgbm_tuned_v2.joblib")
CALIBRATOR_PATH = os.path.join(THIS_DIR, "..", "models", "isotonic_calibrator_v2.joblib")

app = FastAPI(title="FluxRoute Smart Payment Routing")

_artifact = joblib.load(MODEL_PATH)
_model = _artifact["model"]
_preprocessor = _artifact["preprocessor"]
_calibrator = joblib.load(CALIBRATOR_PATH)


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
    transaction_id = request.transaction_id or str(uuid.uuid4())

    eligible_routes = get_eligible_routes(request.payment_method, request.network)

    if not eligible_routes:
        raise HTTPException(
            status_code=422,
            detail=(
                f"No route supports payment_method={request.payment_method} "
                f"network={request.network}"
            ),
        )

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
        )
        for route_id in eligible_routes
    ]

    candidates_df = pd.DataFrame(rows)

    X = candidates_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    raw_probabilities = _model.predict_proba(_preprocessor.transform(X))[:, 1]
    candidates_df["raw_success_probability"] = raw_probabilities
    candidates_df["predicted_success_probability"] = _calibrator.predict(raw_probabilities)

    best_index = candidates_df["raw_success_probability"].idxmax()
    chosen_route_id = candidates_df.loc[best_index, "route_id"]
    chosen_probability = float(candidates_df.loc[best_index, "predicted_success_probability"]) # type: ignore

    record_routing_decision(
        route_id=chosen_route_id, # type: ignore
        bank=request.bank,
        network=request.network,
        amount=request.amount,
    )

    record_pending_route(transaction_id, chosen_route_id, request.bank, request.network) # type: ignore

    return RouteResponse(
        transaction_id=transaction_id,
        chosen_route_id=chosen_route_id, # type: ignore
        predicted_success_probability=chosen_probability,
        candidates=[
            CandidateScore(
                route_id=row["route_id"],
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
            detail=f"No pending routing decision found for transaction_id={request.transaction_id}",
        )

    record_transaction_outcome(
        route_id=pending["route_id"],
        bank=pending["bank"],
        network=pending["network"],
        success=request.success,
        latency_ms=request.latency_ms,
    )

    return {"status": "recorded", "transaction_id": request.transaction_id}


@app.get("/health")
def health_check() -> dict:
    try:
        get_redis_client().ping()
        redis_ok = True
    except Exception:
        redis_ok = False

    return {"status": "ok" if redis_ok else "degraded", "redis_connected": redis_ok}