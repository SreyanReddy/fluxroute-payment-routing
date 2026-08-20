from feature_extraction3 import build_features
from simulator2 import run_actual_simulation

SIMULATION_DAYS = 28
SIMULATION_MINUTES = SIMULATION_DAYS * 24 * 60

RAW_PATH = "payment_dataset_actual_4week_v2.csv"
ROUTE_METRICS_PATH = "route_metrics_4week_v2.csv"
FEATURES_PATH = "payment_dataset_features_4week_v2.csv"

if __name__ == "__main__":
    raw_df, route_metrics_df = run_actual_simulation(
        simulation_minutes=SIMULATION_MINUTES
    )

    raw_df.to_csv(RAW_PATH, index=False)
    route_metrics_df.to_csv(
        ROUTE_METRICS_PATH,
        index=False,
    )

    features_df = build_features(
        input_path=RAW_PATH,
        route_metrics_path=ROUTE_METRICS_PATH,
        output_path=FEATURES_PATH,
        train_fraction=0.75,
    )

    print("\nRaw sampled rows:", len(raw_df))
    print("Route-metric rows:", len(route_metrics_df))
    print("Feature rows:", len(features_df))