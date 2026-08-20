import numpy as np
import pandas as pd

N_MINUTES = 1440 
ROUTES = ["R1", "R2", "R3", "R4"]

WINDOW_5M = 5
WINDOW_15M = 15
WINDOW_LOAD = 2            
WINDOW_1M = 1              
FLAG_DOWN_THRESHOLD = 0.70  
NO_FAILURE_SENTINEL = 9999  
BANK_ROUTE_SHRINK_K = 10    

ROUTE_CONFIG = {
    "R1": {"route_cost_percent": 2.1},
    "R2": {"route_cost_percent": 1.45},
    "R3": {"route_cost_percent": 1.1},
    "R4": {"route_cost_percent": 1.75},
}


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    assert df["route_id"].isin(ROUTES).all(), "unexpected route_id values -- check ROUTES list"
    return df


def prior_rolling_sum(values: np.ndarray, window: int) -> np.ndarray:
    n = len(values)
    cumsum = np.concatenate(([0.0], np.cumsum(values)))
    lo = np.clip(np.arange(n) - window, 0, None)
    return cumsum[np.arange(n)] - cumsum[lo]


def per_minute_table(df: pd.DataFrame, group_cols: list, minute_col="minute") -> dict:
    agg = (
        df.groupby(group_cols + [minute_col])
        .agg(count=("success", "size"), success_sum=("success", "sum"), latency_sum=("latency_ms", "sum"))
        .reset_index()
    )
    tables = {}
    for key, g in agg.groupby(group_cols):
        key = key if isinstance(key, tuple) else (key,)
        full = pd.DataFrame({minute_col: np.arange(N_MINUTES)})
        full = full.merge(g, on=minute_col, how="left").fillna(0)
        tables[key] = full
    return tables


def compute_route_level_features(df: pd.DataFrame) -> pd.DataFrame:
    tables = per_minute_table(df, ["route_id"])
    rows = []
    for route in ROUTES:
        t = tables.get((route,))
        if t is None:
            continue
        count = t["count"].to_numpy()
        succ = t["success_sum"].to_numpy()
        lat = t["latency_sum"].to_numpy()

        c5 = prior_rolling_sum(count, WINDOW_5M)
        s5 = prior_rolling_sum(succ, WINDOW_5M)
        l5 = prior_rolling_sum(lat, WINDOW_5M)
        c15 = prior_rolling_sum(count, WINDOW_15M)
        s15 = prior_rolling_sum(succ, WINDOW_15M)
        c_load = prior_rolling_sum(count, WINDOW_LOAD)
        c_1m = prior_rolling_sum(count, WINDOW_1M)

        with np.errstate(invalid="ignore", divide="ignore"):
            rate_5m = np.where(c5 > 0, s5 / c5, np.nan)
            rate_15m = np.where(c15 > 0, s15 / c15, np.nan)
            avg_lat_5m = np.where(c5 > 0, l5 / c5, np.nan)

        had_failure = (count - succ) > 0
        tsf = np.full(N_MINUTES, NO_FAILURE_SENTINEL, dtype=float)
        last_failure_minute = None
        for m in range(N_MINUTES):
            if last_failure_minute is not None:
                tsf[m] = m - last_failure_minute
            if had_failure[m]:
                last_failure_minute = m

        flagged_down = (rate_5m < FLAG_DOWN_THRESHOLD).astype(float)
        flagged_down[np.isnan(rate_5m)] = 0 

        rows.append(pd.DataFrame({
            "route_id": route,
            "minute": np.arange(N_MINUTES),
            "route_success_rate_5m": rate_5m,
            "route_success_rate_15m": rate_15m,
            "route_avg_latency_5m": avg_lat_5m,
            "route_current_load": c_load,
            "route_requests_1m": c_1m,
            "time_since_last_failure": tsf,
            "route_flagged_down": flagged_down,
        }))
    return pd.concat(rows, ignore_index=True)


def compute_bank_route_feature(df: pd.DataFrame, route_base_success_rate: dict) -> pd.DataFrame:
    tables = per_minute_table(df, ["bank", "route_id"])
    rows = []
    for (bank, route), t in tables.items():
        count = t["count"].to_numpy()
        succ = t["success_sum"].to_numpy()
        c15 = prior_rolling_sum(count, WINDOW_15M)
        s15 = prior_rolling_sum(succ, WINDOW_15M)

        prior = route_base_success_rate.get(route, np.nanmean(list(route_base_success_rate.values())))
        shrunk = (s15 + BANK_ROUTE_SHRINK_K * prior) / (c15 + BANK_ROUTE_SHRINK_K)

        rows.append(pd.DataFrame({
            "bank": bank,
            "route_id": route,
            "minute": np.arange(N_MINUTES),
            "bank_route_success_rate_15m": shrunk,
        }))
    return pd.concat(rows, ignore_index=True)


def build_features(path_in: str, path_out: str):
    df = load_data(path_in)

    base_stats = df.groupby("route_id").agg(
        route_base_success_rate=("success", "mean"),
        route_base_latency_ms=("latency_ms", "mean"),
    )
    route_base_success_rate = base_stats["route_base_success_rate"].to_dict()

    route_feats = compute_route_level_features(df)
    bank_route_feats = compute_bank_route_feature(df, route_base_success_rate)

    out = df.merge(route_feats, on=["route_id", "minute"], how="left")
    out = out.merge(bank_route_feats, on=["bank", "route_id", "minute"], how="left")
    out = out.merge(base_stats, on="route_id", how="left")
    out["route_cost_percent"] = out["route_id"].map(lambda r: ROUTE_CONFIG[r]["route_cost_percent"])

    final_cols = [
        "txn_id", "minute",
        # transaction
        "amount", "bank", "network", "payment_method", "merchant", "device", "risk",
        # route (static)
        "route_id", "route_base_success_rate", "route_base_latency_ms", "route_cost_percent",
        # real-time route state
        "route_success_rate_5m", "route_success_rate_15m", "route_avg_latency_5m",
        "route_current_load", "route_requests_1m", "time_since_last_failure", "route_flagged_down",
        # interaction/history
        "bank_route_success_rate_15m",
        # label
        "success",
    ]
    out = out[final_cols].sort_values(["minute", "txn_id"]).reset_index(drop=True)
    out.to_csv(path_out, index=False)
    return out


if __name__ == "__main__":
    result = build_features(
        "./payment_dataset_actual_1day.csv",
        "./payment_dataset_features_1day.csv",
    )
    print("Shape:", result.shape)
    print()
    print("Nulls per column:")
    print(result.isnull().sum())
    print()
    print(result.head())