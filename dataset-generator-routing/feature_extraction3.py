import numpy as np
import pandas as pd

MINUTES_PER_DAY = 24 * 60

TRAIN_FRACTION = 0.75

ROUTES = ["R1", "R2", "R3", "R4"]

WINDOW_5M = 5
WINDOW_15M = 15
WINDOW_LOAD = 2
WINDOW_1M = 1

FLAG_DOWN_THRESHOLD = 0.70
NO_FAILURE_SENTINEL = 9999
BANK_ROUTE_SHRINK_K = 10


ROUTE_CONFIG = {
    "R1": {"route_cost_percent": 2.10},
    "R2": {"route_cost_percent": 1.45},
    "R3": {"route_cost_percent": 1.10},
    "R4": {"route_cost_percent": 1.75},
}


def load_data(path: str) -> pd.DataFrame:
    """Load the transaction dataset and validate basic fields."""

    df = pd.read_csv(path)

    if not df["route_id"].isin(ROUTES).all():
        raise ValueError("Dataset contains an unknown route_id.")

    if df["minute"].min() < 0:
        raise ValueError("minute cannot be negative.")

    return df


# ============================================================
# Rolling-window helpers
# ============================================================

def prior_rolling_sum(values: np.ndarray, window: int) -> np.ndarray:
    """
    Sum the previous `window` minutes.

    The current minute is NOT included.

    Example with window=5 at minute 10:
        uses minutes 5, 6, 7, 8, 9

    This keeps all features strictly historical and prevents
    the current transaction's outcome from leaking into its features.
    """

    cumulative = np.concatenate(([0.0], np.cumsum(values)))

    current = np.arange(len(values))
    start = np.maximum(current - window, 0)

    return cumulative[current] - cumulative[start]


# ============================================================
# Per-minute aggregation (sampled-transaction level only --
# used for bank/route features, not route-level health/capacity)
# ============================================================

def build_minute_table(
    df: pd.DataFrame,
    group_columns: list[str],
    total_minutes: int,
) -> dict:
    """
    Aggregate transactions by group and minute.

    Missing minutes are filled with zero so every group has a
    continuous timeline from minute 0 to the end of the dataset.
    """

    aggregated = (
        df.groupby(group_columns + ["minute"])
        .agg(
            request_count=("success", "size"),
            success_count=("success", "sum"),
            latency_sum=("latency_ms", "sum"),
        )
        .reset_index()
    )

    tables = {}

    for key, group in aggregated.groupby(group_columns):

        if not isinstance(key, tuple):
            key = (key,)

        timeline = pd.DataFrame({
            "minute": np.arange(total_minutes)
        })

        timeline = timeline.merge(
            group,
            on="minute",
            how="left",
        )

        timeline[
            ["request_count", "success_count", "latency_sum"]
        ] = timeline[
            ["request_count", "success_count", "latency_sum"]
        ].fillna(0)

        tables[key] = timeline

    return tables


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    minute_of_day = df["minute"] % MINUTES_PER_DAY
    df["day_in_week"] = (df["minute"] // MINUTES_PER_DAY) % 7
    df["hour_of_day"] = (minute_of_day // 60)

    return df


# ============================================================
# Route-level features -- now sourced from aggregate route_metrics,
# not the small sampled-transaction dataframe. route_metrics already
# has one row per (route_id, minute) with no gaps, so no reindexing
# is needed here (unlike build_minute_table above).
# ============================================================

def calculate_route_features(
    route_metrics: pd.DataFrame,
    total_minutes: int,
) -> pd.DataFrame:
    """
    Calculate historical route-level features from aggregate,
    production-scale traffic (see simulator.py::build_route_metrics).

    IMPORTANT:
    The timeline is continuous across all days.
    Rolling windows do NOT reset at midnight.
    """

    feature_frames = []

    for route in ROUTES:

        table = (
            route_metrics[route_metrics["route_id"] == route]
            .sort_values("minute")
            .reset_index(drop=True)
        )

        if table.empty:
            continue

        requests = table["aggregate_request_count"].to_numpy()
        successes = table["aggregate_success_count"].to_numpy()
        latency = table["aggregate_latency_sum"].to_numpy()

        # ----------------------------
        # Historical windows
        # ----------------------------

        requests_5m = prior_rolling_sum(requests, WINDOW_5M)
        successes_5m = prior_rolling_sum(successes, WINDOW_5M)
        latency_5m = prior_rolling_sum(latency, WINDOW_5M)

        requests_15m = prior_rolling_sum(requests, WINDOW_15M)
        successes_15m = prior_rolling_sum(successes, WINDOW_15M)

        requests_load = prior_rolling_sum(requests, WINDOW_LOAD)
        requests_1m = prior_rolling_sum(requests, WINDOW_1M)

        # ----------------------------
        # Success rates
        # ----------------------------

        with np.errstate(divide="ignore", invalid="ignore"):

            success_rate_5m = np.where(
                requests_5m > 0,
                successes_5m / requests_5m,
                np.nan,
            )

            success_rate_15m = np.where(
                requests_15m > 0,
                successes_15m / requests_15m,
                np.nan,
            )

            avg_latency_5m = np.where(
                requests_5m > 0,
                latency_5m / requests_5m,
                np.nan,
            )

        # ----------------------------
        # Time since last incident
        #
        # Keyed off health_state, not raw aggregate failure counts.
        # At production scale, some failure occurs in almost every
        # single minute regardless of route health, so counting
        # aggregate failures directly is meaningless here.
        # ----------------------------

        # had_incident = table["health_state"].to_numpy() != "HEALTHY"
        had_incident = success_rate_5m < FLAG_DOWN_THRESHOLD
        had_incident = np.where(np.isnan(success_rate_5m), False, had_incident)
        
        time_since_failure = np.full(
            len(table),
            NO_FAILURE_SENTINEL,
            dtype=float,
        )

        last_incident_minute = None

        for i in range(len(table)):
            if last_incident_minute is not None:
                time_since_failure[i] = i - last_incident_minute
            if had_incident[i]:
                last_incident_minute = i

        # ----------------------------
        # Route health flag
        # ----------------------------

        flagged_down = (
            success_rate_5m < FLAG_DOWN_THRESHOLD
        ).astype(float)

        # No historical traffic means we cannot say
        # the route is down.
        flagged_down[np.isnan(success_rate_5m)] = 0

        feature_frames.append(
            pd.DataFrame({
                "route_id": route,
                "minute": table["minute"].to_numpy(),

                "route_success_rate_5m": success_rate_5m,
                "route_success_rate_15m": success_rate_15m,
                "route_avg_latency_5m": avg_latency_5m,

                "route_current_load": requests_load,
                "route_requests_1m": requests_1m,

                "time_since_last_failure": time_since_failure,

                "route_flagged_down": flagged_down,

                "route_utilization": table["route_utilization"].to_numpy(),
                "route_effective_capacity_tps": (
                    table["route_effective_capacity_tps"].to_numpy()
                ),
                "system_tps": table["system_tps"].to_numpy(),
            })
        )

    return pd.concat(
        feature_frames,
        ignore_index=True,
    )


# ============================================================
# Bank + route historical feature (sampled-transaction level --
# no aggregate equivalent exists, unchanged from before)
# ============================================================

def calculate_bank_route_features(
    df: pd.DataFrame,
    route_baselines: dict,
    total_minutes: int,
) -> pd.DataFrame:
    """
    Calculate bank + route success rate using a smoothed
    historical 15-minute success rate.

    Bayesian-style shrinkage prevents low-volume bank/route
    combinations from producing unstable success rates.
    """

    minute_tables = build_minute_table(
        df,
        ["bank", "route_id"],
        total_minutes,
    )

    feature_frames = []

    overall_baseline = np.mean(
        list(route_baselines.values())
    )

    for (bank, route), table in minute_tables.items():

        requests = table["request_count"].to_numpy()
        successes = table["success_count"].to_numpy()

        requests_15m = prior_rolling_sum(
            requests,
            WINDOW_15M,
        )

        successes_15m = prior_rolling_sum(
            successes,
            WINDOW_15M,
        )

        prior_success_rate = route_baselines.get(
            route,
            overall_baseline,
        )

        smoothed_rate = (
            successes_15m
            + BANK_ROUTE_SHRINK_K * prior_success_rate
        ) / (
            requests_15m
            + BANK_ROUTE_SHRINK_K
        )

        feature_frames.append(
            pd.DataFrame({
                "bank": bank,
                "route_id": route,
                "minute": np.arange(total_minutes),
                "bank_route_success_rate_15m": smoothed_rate,
            })
        )

    return pd.concat(
        feature_frames,
        ignore_index=True,
    )


# ============================================================
# Training-only baselines
# ============================================================

def calculate_training_baselines(
    df: pd.DataFrame,
    train_minutes: int,
) -> pd.DataFrame:
    """
    Calculate route baseline statistics using ONLY the
    training window.

    This prevents the test period from leaking into features.
    """

    training_data = df[
        df["minute"] < train_minutes
    ]

    return (
        training_data
        .groupby("route_id")
        .agg(
            route_base_success_rate=("success", "mean"),
            route_base_latency_ms=("latency_ms", "mean"),
        )
        .reset_index()
    )


def build_features(
    input_path: str,
    route_metrics_path: str,
    output_path: str,
    train_fraction: float = TRAIN_FRACTION,
) -> pd.DataFrame:

    # Load data

    df = load_data(input_path)
    route_metrics = pd.read_csv(route_metrics_path)

    total_minutes = int(df["minute"].max()) + 1
    train_minutes = int(total_minutes * train_fraction)

    df = add_time_features(df)

    print(f"Loaded {len(df):,} sampled transactions.")
    print(f"Route-metric rows: {len(route_metrics):,}")
    print(f"Timeline: {total_minutes:,} minutes")

    base_stats = calculate_training_baselines(df, train_minutes)

    route_baselines = dict(
        zip(
            base_stats["route_id"],
            base_stats["route_base_success_rate"],
        )
    )

    route_features = calculate_route_features(
        route_metrics,
        total_minutes,
    )

    # Historical bank-route features

    bank_route_features = calculate_bank_route_features(
        df,
        route_baselines,
        total_minutes,
    )

    # Merge features

    result = df.merge(route_features, on=["route_id", "minute"], how="left")

    result = result.merge(bank_route_features, on=["bank", "route_id", "minute"], how="left")

    # Add training-derived baselines

    result = result.merge(base_stats, on="route_id", how="left")

    # Static route cost

    result["route_cost_percent"] = (
        result["route_id"]
        .map(lambda route:
             ROUTE_CONFIG[route]["route_cost_percent"]
        )
    )

    # Train / test split

    result["dataset_split"] = np.where(result["minute"] < train_minutes, "train", "test")

    final_columns = [
        "txn_id",
        "minute",
        "day_in_week",
        "hour_of_day",

        # Transaction
        "amount",
        "bank",
        "network",
        "payment_method",
        "merchant",
        "device",
        "risk",

        # Route
        "route_id",
        "route_base_success_rate",
        "route_base_latency_ms",
        "route_cost_percent",

        # Real-time route state
        "route_success_rate_5m",
        "route_success_rate_15m",
        "route_avg_latency_5m",
        "route_current_load",
        "route_requests_1m",
        "time_since_last_failure",
        "route_flagged_down",
        "route_utilization",
        "route_effective_capacity_tps",
        "system_tps",

        # Bank + route history
        "bank_route_success_rate_15m",

        # Target
        "success",

        # Dataset split
        "dataset_split",
    ]
    # success_probability / health_state are deliberately excluded --
    # they'd leak the label-generating probability directly.

    result = (result[final_columns].sort_values(["minute", "txn_id"]).reset_index(drop=True))

    result.to_csv(output_path, index=False)

    return result


if __name__ == "__main__":

    result = build_features(
        input_path="./payment_dataset_actual_4week_v2.csv",
        route_metrics_path="./route_metrics_4week_v2.csv",
        output_path="./payment_dataset_features_4week_v2.csv",
    )

    print("\nFeature dataset created.")
    print("Shape:", result.shape)

    print("\nTrain / test split:")
    print(result["dataset_split"].value_counts())

    print("\nNulls per column:")
    print(result.isnull().sum())

    print("\nFirst rows:")
    print(result.head())