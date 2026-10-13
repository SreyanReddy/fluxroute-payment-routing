"""
Rebuilds "what the world looked like just before a transaction".

To compute features for a transaction at minute m, the serving code needs
history: how its route, bank and network performed in the minutes before m,
plus a few running totals. Training computed its features from exactly this
history. This file reconstructs it from the same two CSVs training used, so
the serving code can be handed identical inputs.

Training used two different streams, so the history keeps them separate:

    aggregate stream   route_metrics_4week_v2.csv
        Production-scale traffic. Training used it for route health
        (success rate 5m/15m, average latency, incidents) and system_tps.

    sampled stream     payment_dataset_actual_4week_v2.csv
        The individual transactions. Training used it for bank, network and
        bank x route features, the amount ratio, last-success time, and
        route_failure_rate_1m.

Nothing in here knows how the serving code stores things. That is the
adapter's job.
"""

import sys
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "dataset-generator-routing"))

from feature_extraction3 import (  # noqa: E402  (training's own helpers)
    FLAG_DOWN_THRESHOLD,
    WINDOW_5M,
    prior_rolling_sum,
)

DATA_DIR = PROJECT_ROOT / "dataset-generator-routing"
SAMPLED_CSV = DATA_DIR / "payment_dataset_actual_4week_v2.csv"
AGGREGATE_CSV = DATA_DIR / "route_metrics_4week_v2.csv"

# How far back to rebuild history.
#   Clock-window features (route health, system traffic, bank x route) never
#   read further back than 15 minutes.
#   Bank/network failure rates look at the last 5 minutes that had any
#   traffic; for a quiet bank at night that can be hours ago.
CLOCK_LOOKBACK_MINUTES = 15
ACTIVE_LOOKBACK_MINUTES = 6 * 60


class Counts(NamedTuple):
    attempts: int
    successes: int


class RouteMinute(NamedTuple):
    attempts: int
    successes: int
    latency_sum: float


@dataclass
class MomentHistory:
    """Everything the serving code is allowed to know just before `minute`."""

    minute: int
    route_id: str
    bank: str
    network: str

    route_minutes: dict[int, RouteMinute]      # aggregate stream, this route
    system_requests: dict[int, int]            # aggregate stream, all routes
    route_sampled_minutes: dict[int, Counts]   # sampled stream, this route
    bank_minutes: dict[int, Counts]            # sampled stream
    network_minutes: dict[int, Counts]         # sampled stream
    bank_route_minutes: dict[int, Counts]      # sampled stream

    last_flagged_minute: int | None            # last route incident before m
    last_bank_route_success_minute: int | None
    bank_amount_sum: float                     # earlier transactions only
    bank_amount_count: int


class HistoryBuilder:
    """Loads both CSVs once, then answers history_before(txn) quickly."""

    def __init__(self):
        sampled = pd.read_csv(
            SAMPLED_CSV,
            usecols=["txn_id", "minute", "amount", "bank", "network", "route_id", "success"],
        )
        aggregate = pd.read_csv(
            AGGREGATE_CSV,
            usecols=[
                "minute",
                "route_id",
                "aggregate_request_count",
                "aggregate_success_count",
                "aggregate_latency_sum",
            ],
        )

        self._route_minutes = {
            (row.route_id, row.minute): RouteMinute(
                int(row.aggregate_request_count),
                int(row.aggregate_success_count),
                float(row.aggregate_latency_sum),
            )
            for row in aggregate.itertuples(index=False)
        }
        self._system_requests = (
            aggregate.groupby("minute")["aggregate_request_count"].sum().astype(int).to_dict()
        )

        self._route_sampled = _counts_per_minute(sampled, ["route_id"])
        self._bank = _counts_per_minute(sampled, ["bank"])
        self._network = _counts_per_minute(sampled, ["network"])
        self._bank_route = _counts_per_minute(sampled, ["bank", "route_id"])

        self._incident_minutes = _incident_minutes_per_route(aggregate)
        self._earlier_state = _earlier_state_per_transaction(sampled)

    def history_before(self, txn) -> MomentHistory:
        m = int(txn.minute)
        route, bank, network = txn.route_id, txn.bank, txn.network
        clock_window = range(m - CLOCK_LOOKBACK_MINUTES, m)  # strictly before m
        active_window = range(m - ACTIVE_LOOKBACK_MINUTES, m)

        earlier_sum, earlier_count, last_success = self._earlier_state[int(txn.txn_id)]

        return MomentHistory(
            minute=m,
            route_id=route,
            bank=bank,
            network=network,
            route_minutes=_pick(self._route_minutes, (route,), clock_window),
            system_requests={k: self._system_requests[k] for k in clock_window if k in self._system_requests},
            route_sampled_minutes=_pick(self._route_sampled, (route,), active_window),
            bank_minutes=_pick(self._bank, (bank,), active_window),
            network_minutes=_pick(self._network, (network,), active_window),
            bank_route_minutes=_pick(self._bank_route, (bank, route), clock_window),
            last_flagged_minute=self._last_incident_before(route, m),
            last_bank_route_success_minute=last_success,
            bank_amount_sum=earlier_sum,
            bank_amount_count=earlier_count,
        )

    def _last_incident_before(self, route: str, m: int) -> int | None:
        minutes = self._incident_minutes[route]
        i = bisect_left(minutes, m)  # minutes[i-1] is the last incident before m
        return minutes[i - 1] if i > 0 else None


def _pick(table: dict, key_prefix: tuple, window: range) -> dict:
    """Pull the minutes in `window` for one key out of a lookup table."""
    out = {}
    for minute in window:
        value = table.get(key_prefix + (minute,))
        if value is not None:
            out[minute] = value
    return out


def _counts_per_minute(sampled: pd.DataFrame, keys: list[str]) -> dict:
    """{(key..., minute): Counts} for every minute that had transactions."""
    grouped = sampled.groupby(keys + ["minute"])["success"].agg(["size", "sum"])
    return {
        key: Counts(int(attempts), int(successes))
        for key, (attempts, successes) in zip(grouped.index, grouped.to_numpy())
    }


def _incident_minutes_per_route(aggregate: pd.DataFrame) -> dict[str, list[int]]:
    """
    The spec's incident rule: minute k is an incident for a route when that
    route's 5-minute success rate (over minutes k-5 .. k-1) is below 0.70.
    Uses training's own prior_rolling_sum so the window is identical.
    """
    incidents = {}
    for route, table in aggregate.groupby("route_id"):
        table = table.sort_values("minute")
        minutes = table["minute"].to_numpy()
        assert (minutes == np.arange(len(minutes))).all(), "route_metrics must have every minute"

        requests_5m = prior_rolling_sum(table["aggregate_request_count"].to_numpy(dtype=float), WINDOW_5M)
        successes_5m = prior_rolling_sum(table["aggregate_success_count"].to_numpy(dtype=float), WINDOW_5M)
        with np.errstate(divide="ignore", invalid="ignore"):
            rate_5m = np.where(requests_5m > 0, successes_5m / requests_5m, np.nan)

        is_incident = (rate_5m < FLAG_DOWN_THRESHOLD) & ~np.isnan(rate_5m)
        incidents[route] = minutes[is_incident].tolist()
    return incidents


def _earlier_state_per_transaction(sampled: pd.DataFrame) -> dict:
    """
    For each transaction, the running state built from transactions that came
    *before it* (in txn_id order, the order training used):
        - sum and count of earlier amounts at the same bank
        - minute of the most recent earlier success on the same bank + route
    """
    s = sampled.sort_values("txn_id")

    running_sum = s.groupby("bank")["amount"].cumsum()
    earlier_sum = running_sum.groupby(s["bank"]).shift(1).fillna(0.0)
    earlier_count = s.groupby("bank").cumcount()

    success_minute = s["minute"].where(s["success"] == 1)
    pair = [s["bank"], s["route_id"]]
    latest_success = success_minute.groupby(pair).ffill()
    earlier_success = latest_success.groupby(pair).shift(1)

    return {
        int(txn_id): (float(total), int(count), None if pd.isna(last) else int(last))
        for txn_id, total, count, last in zip(s["txn_id"], earlier_sum, earlier_count, earlier_success)
    }
