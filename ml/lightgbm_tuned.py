import os
import joblib
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from lightgbm import LGBMClassifier, early_stopping

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    log_loss,
    brier_score_loss,
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    precision_recall_curve,
)

from sklearn.model_selection import ParameterSampler

DATA_PATH = "./dataset-generator-routing/payment_dataset_features_4week_v2.csv"

MODEL_PATH = "./models/lightgbm_tuned_v2.joblib"

TARGET = "success"

RANDOM_STATE = 42

N_SEARCH_ITERATIONS = 20

TARGET_FAILURE_RECALL = 0.50

# We care about detecting failures (class 0).
# This controls how much more heavily failures are weighted
# during model training.
USE_CLASS_WEIGHTING = True


DROP_COLUMNS = [
    "txn_id",
    "dataset_split",
]

NUMERIC_FEATURES = [
    "day_in_week",
    "hour_of_day",
    "amount",

    "route_base_success_rate",
    "route_base_latency_ms",
    "route_cost_percent",

    "route_success_rate_5m",
    "route_success_rate_15m",
    "route_avg_latency_5m",

    "route_utilization",
    "system_tps",

    "time_since_last_failure",
    "route_flagged_down",

    "bank_route_success_rate_15m",

    "route_failure_rate_5m",
    "route_failure_rate_15m",

    "route_success_drop_5m",
    "route_success_drop_15m",

    "route_latency_ratio",

    "bank_route_gap",

    "route_stress",

    "bank_failure_rate_5m",
    "network_failure_rate_5m",
    "route_failure_rate_1m",
    "time_since_last_bank_route_success",
    "amount_to_bank_avg_ratio",
]


CATEGORICAL_FEATURES = [
    "bank",
    "network",
    "payment_method",
    "merchant",
    "device",
    "risk",
    "route_id",
]


# ============================================================
# COST ASSUMPTIONS (for hyperparameter selection)
# ============================================================

FAILURE_COST_FIXED = 50.0
FAILURE_COST_PCT_OF_AMOUNT = 0.02
COST_THRESHOLD_SWEEP = np.arange(0.02, 0.55, 0.02)


def compute_best_validation_cost(probabilities, y_true, amount, route_fee_pct):
    """
    Sweeps thresholds and returns the best (minimum) achievable cost
    per transaction for this candidate model on validation data.
    """
    y_true = np.asarray(y_true)
    best_cost = float("inf")

    for threshold in COST_THRESHOLD_SWEEP:
        failure_probs = 1.0 - probabilities
        predicted_failure = failure_probs >= threshold

        actual_failure = (y_true == 0)
        fn_mask = actual_failure & ~predicted_failure
        fp_mask = ~actual_failure & predicted_failure
        tn_mask = ~actual_failure & ~predicted_failure

        route_fee = amount * route_fee_pct
        cost = (
            fn_mask.sum() * FAILURE_COST_FIXED
            + (amount[fn_mask] * FAILURE_COST_PCT_OF_AMOUNT).sum()
            + route_fee[fp_mask].sum()
            - route_fee[tn_mask].sum()
        )

        cost_per_txn = cost / len(y_true)
        if cost_per_txn < best_cost:
            best_cost = cost_per_txn

    return best_cost

# ============================================================
# FEATURE ENGINEERING
# ============================================================

# ============================================================
# 1. BANK-LEVEL FAILURE RATE — 5 MINUTES
# ============================================================

def add_bank_failure_rate_5m(df):
    """
    Calculates the failure rate of each bank across ALL routes
    over the previous 5 minutes.

    Current transaction is excluded to prevent leakage.
    """

    df = df.sort_values("txn_id").copy()

    # Failure = 1 when success == 0
    failure = 1 - df["success"]

    # Group by bank + minute and aggregate failures / transactions.
    bank_minute = (
        df.groupby(["bank", "minute"])
        .agg(
            bank_failures=("success", lambda x: (x == 0).sum()),
            bank_transactions=("success", "size")
        )
        .reset_index()
    )

    # Rolling 5-minute window, excluding current minute.
    bank_minute["bank_failures_5m"] = (
        bank_minute
        .groupby("bank")["bank_failures"]
        .transform(
            lambda x: x.shift(1).rolling(5, min_periods=1).sum()
        )
    )

    bank_minute["bank_transactions_5m"] = (
        bank_minute
        .groupby("bank")["bank_transactions"]
        .transform(
            lambda x: x.shift(1).rolling(5, min_periods=1).sum()
        )
    )

    bank_minute["bank_failure_rate_5m"] = (
        bank_minute["bank_failures_5m"] /
        bank_minute["bank_transactions_5m"].replace(0, np.nan)
    )

    df = df.merge(
        bank_minute[
            ["bank", "minute", "bank_failure_rate_5m"]
        ],
        on=["bank", "minute"],
        how="left"
    )

    return df


# ============================================================
# 2. NETWORK-LEVEL FAILURE RATE — 5 MINUTES
# ============================================================

def add_network_failure_rate_5m(df):
    """
    Calculates failure rate for the entire payment network
    across all banks and routes over the previous 5 minutes.

    Example:
        UPI failure rate across SBI/HDFC/ICICI/etc.
    """

    df = df.sort_values("txn_id").copy()

    network_minute = (
        df.groupby(["network", "minute"])
        .agg(
            network_failures=("success", lambda x: (x == 0).sum()),
            network_transactions=("success", "size")
        )
        .reset_index()
    )

    network_minute["network_failures_5m"] = (
        network_minute
        .groupby("network")["network_failures"]
        .transform(
            lambda x: x.shift(1).rolling(5, min_periods=1).sum()
        )
    )

    network_minute["network_transactions_5m"] = (
        network_minute
        .groupby("network")["network_transactions"]
        .transform(
            lambda x: x.shift(1).rolling(5, min_periods=1).sum()
        )
    )

    network_minute["network_failure_rate_5m"] = (
        network_minute["network_failures_5m"] /
        network_minute["network_transactions_5m"].replace(0, np.nan)
    )

    df = df.merge(
        network_minute[
            ["network", "minute", "network_failure_rate_5m"]
        ],
        on=["network", "minute"],
        how="left"
    )

    return df


# ============================================================
# 3. ROUTE FAILURE RATE — 1 MINUTE
# ============================================================

def add_route_failure_rate_1m(df):
    """
    Calculates the route's failure rate during the immediately
    preceding minute.

    This gives us a very short-term outage signal.
    """

    df = df.sort_values("txn_id").copy()

    route_minute = (
        df.groupby(["route_id", "minute"])
        .agg(
            route_failures=("success", lambda x: (x == 0).sum()),
            route_transactions=("success", "size")
        )
        .reset_index()
    )

    # Previous minute only.
    route_minute["route_failure_rate_1m"] = (
        route_minute
        .groupby("route_id")["route_failures"]
        .shift(1)
        /
        route_minute
        .groupby("route_id")["route_transactions"]
        .shift(1)
    )

    df = df.merge(
        route_minute[
            ["route_id", "minute", "route_failure_rate_1m"]
        ],
        on=["route_id", "minute"],
        how="left"
    )

    return df


# ============================================================
# 4. TIME SINCE LAST BANK + ROUTE SUCCESS
# ============================================================

def add_time_since_last_bank_route_success(df):
    """
    Calculates time elapsed since the previous successful
    transaction for the same (bank, route_id) pair.

    Example:

        SBI + R2
        last success at minute 100
        current transaction at minute 108

        -> 8 minutes
    """

    df = df.sort_values("txn_id").copy()

    # Time of successful transactions only.
    success_minute = np.where(
        df["success"] == 1,
        df["minute"],
        np.nan
    )

    # Forward-fill the previous successful minute
    # within each bank-route pair.
    df["_last_success_minute"] = (
        pd.Series(success_minute, index=df.index)
        .groupby(
            [df["bank"], df["route_id"]]
        )
        .ffill()
    )

    # IMPORTANT:
    # The current transaction must not see its own success.
    df["_last_success_minute"] = (
        df.groupby(["bank", "route_id"])["_last_success_minute"]
        .shift(1)
    )

    df["time_since_last_bank_route_success"] = (
        df["minute"] - df["_last_success_minute"]
    )

    # Same convention as your existing feature.
    # 9999 = no previous success observed.
    df["time_since_last_bank_route_success"] = (
        df["time_since_last_bank_route_success"]
        .fillna(9999.0)
    )

    df.drop(columns=["_last_success_minute"], inplace=True)

    return df


# ============================================================
# 5. AMOUNT / HISTORICAL BANK AVERAGE
# ============================================================

def add_amount_to_bank_avg_ratio(df):
    """
    Calculates:

        current amount /
        historical average transaction amount for the bank

    Only previous transactions are used.
    """

    df = df.sort_values("txn_id").copy()

    # Historical cumulative amount and count.
    df["_bank_amount_sum"] = (
        df.groupby("bank")["amount"]
        .transform(lambda x: x.cumsum().shift(1))
    )

    df["_bank_amount_count"] = (
        df.groupby("bank")["amount"]
        .cumcount()
    )

    df["_historical_bank_avg_amount"] = (
        df["_bank_amount_sum"] /
        df["_bank_amount_count"].replace(0, np.nan)
    )

    df["amount_to_bank_avg_ratio"] = (
        df["amount"] /
        df["_historical_bank_avg_amount"]
    )

    # First transaction(s) have no history.
    df["amount_to_bank_avg_ratio"] = (
        df["amount_to_bank_avg_ratio"]
        .replace([np.inf, -np.inf], np.nan)
    )

    df.drop(
        columns=[
            "_bank_amount_sum",
            "_bank_amount_count",
            "_historical_bank_avg_amount"
        ],
        inplace=True
    )

    return df

def engineer_features(df):

    df = df.copy()

    # --------------------------------------------------------
    # Recent route failure rates
    # --------------------------------------------------------

    df["route_failure_rate_5m"] = (
        1.0 - df["route_success_rate_5m"]
    )

    df["route_failure_rate_15m"] = (
        1.0 - df["route_success_rate_15m"]
    )

    # --------------------------------------------------------
    # How much worse is the route compared to its baseline?
    # --------------------------------------------------------

    df["route_success_drop_5m"] = (
        df["route_base_success_rate"]
        - df["route_success_rate_5m"]
    )

    df["route_success_drop_15m"] = (
        df["route_base_success_rate"]
        - df["route_success_rate_15m"]
    )

    # --------------------------------------------------------
    # Latency degradation
    #
    # > 1.0 means current latency is worse than baseline.
    # --------------------------------------------------------

    df["route_latency_ratio"] = (
        df["route_avg_latency_5m"]
        / (df["route_base_latency_ms"] + 1e-6)
    )

    # --------------------------------------------------------
    # Bank-specific route degradation
    #
    # Positive -> bank performs better than route overall.
    # Negative -> this bank is doing worse on this route.
    # --------------------------------------------------------

    df["bank_route_gap"] = (
        df["bank_route_success_rate_15m"]
        - df["route_success_rate_15m"]
    )

    # --------------------------------------------------------
    # Route stress
    #
    # Combines current load with recent failure rate.
    # High load + high failure rate = highly stressed route.
    # --------------------------------------------------------

    df["route_stress"] = (
        df["route_utilization"]
        * df["route_failure_rate_15m"]
    )

    df = add_bank_failure_rate_5m(df)
    df = add_network_failure_rate_5m(df)
    df = add_route_failure_rate_1m(df)
    df = add_time_since_last_bank_route_success(df)
    df = add_amount_to_bank_avg_ratio(df)

    return df


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    df = pd.read_csv(DATA_PATH)
    df = engineer_features(df)

    train_df = df[
        df["dataset_split"] == "train"
    ].copy()

    test_df = df[
        df["dataset_split"] == "test"
    ].copy()

    # Keep chronological order.
    train_df = train_df.sort_values(
        "minute"
    )

    # Last 20% of training data becomes validation.
    split_index = int(
        len(train_df) * 0.8
    )

    train_part = train_df.iloc[
        :split_index
    ].copy()

    validation_part = train_df.iloc[
        split_index:
    ].copy()

    return (
        train_part,
        validation_part,
        test_df,
    )


# ============================================================
# PREPROCESSING
# ============================================================

def build_preprocessor():

    numeric_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            ),
        ),
    ])

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            ),
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=True,
            ),
        ),
    ])

    return ColumnTransformer([
        (
            "numeric",
            numeric_pipeline,
            NUMERIC_FEATURES,
        ),
        (
            "categorical",
            categorical_pipeline,
            CATEGORICAL_FEATURES,
        ),
    ])


# ============================================================
# PARAMETER SEARCH SPACE
# ============================================================

PARAMETER_SPACE = {

    "num_leaves": [
        15,
        31,
        63,
    ],

    "max_depth": [
        -1,
        5,
        8,
        12,
        16,
    ],

    "learning_rate": [
        0.05,
        0.08,
        0.10,
    ],

    "min_child_samples": [
        20,
        50,
        100,
    ],

    "subsample": [
        0.7,
        0.8,
        0.9,
        1.0,
    ],

    "colsample_bytree": [
        0.7,
        0.8,
        0.9,
        1.0,
    ],

    "reg_alpha": [
        0.1,
        0.5,
        1.0,
    ],

    "reg_lambda": [
        0.0,
        0.01,
        0.1,
        0.5,
        1.0,
    ],

    "class_weight_multiplier": [0.5, 0.75, 1.0, 1.5, 2.0, 3.0],
}


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_xy(df):

    X = df.drop(
        columns=[
            TARGET,
            *DROP_COLUMNS,
        ]
    )

    y = df[TARGET]

    return X, y


# ============================================================
# CLASS WEIGHT
# ============================================================

def calculate_scale_pos_weight(y):

    failure_count = (y == 0).sum()
    success_count = (y == 1).sum()

    print("\nClass distribution:")
    print(f"Failures: {failure_count:,}")
    print(f"Successes: {success_count:,}")

    print(
        f"Failure ratio: "
        f"{failure_count / len(y):.4%}"
    )

    return success_count / failure_count


# ============================================================
# TRAIN ONE MODEL
# ============================================================

def train_candidate(
    params,
    X_train_transformed,
    y_train,
    X_validation_transformed,
    y_validation,
    class_weight,
    val_amount,
    val_route_fee_pct,
):

    model = LGBMClassifier(
        objective="binary",
        n_estimators=3000,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1,
        class_weight=class_weight,
        **params,
    )

    model.fit(
        X_train_transformed,
        y_train,
        eval_X=X_validation_transformed,
        eval_y=y_validation,
        eval_metric="binary_logloss",
        callbacks=[
            early_stopping(
                stopping_rounds=100,
                verbose=False,
            )
        ],
    )

    probabilities = model.predict_proba(
        X_validation_transformed
    )[:, 1] # type: ignore

    cost = compute_best_validation_cost(
        probabilities,
        y_validation,
        val_amount,
        val_route_fee_pct,
    )

    return (model, cost)


# ============================================================
# THRESHOLD ANALYSIS
# ============================================================

def analyze_thresholds(
    model,
    preprocessor,
    validation_df,
):

    X_validation, y_validation = prepare_xy(
        validation_df
    )

    X_validation_transformed = (
        preprocessor.transform(
            X_validation
        )
    )

    # Model outputs P(success)
    success_probabilities = (
        model.predict_proba(
            X_validation_transformed
        )[:, 1]
    )

    # Convert to P(failure)
    failure_probabilities = (
        1.0 - success_probabilities
    )

    print("\n" + "=" * 80)
    print("THRESHOLD ANALYSIS — VALIDATION SET")
    print("=" * 80)

    thresholds_to_check = [
        0.50,
        0.45,
        0.40,
        0.35,
        0.30,
        0.25,
        0.20,
        0.15,
        0.10,
        0.075,
        0.05,
        0.025,
        0.01,
    ]

    print(
        f"\n{'Failure Threshold':>20}"
        f"{'Failure Precision':>20}"
        f"{'Failure Recall':>18}"
        f"{'Failure F1':>15}"
        f"{'Accuracy':>12}"
        f"{'Pred Failures':>16}"
    )

    print("-" * 105)

    for threshold in thresholds_to_check:

        # Predict failure when:
        #
        # P(failure) >= threshold
        #
        predictions = (
            failure_probabilities >= threshold
        ).astype(int)

        # Convert predictions to actual labels:
        #
        # 1 = success
        # 0 = failure
        predictions = 1 - predictions

        failure_precision = precision_score(
            y_validation,
            predictions,
            pos_label=0,
            zero_division=0,
        )

        failure_recall = recall_score(
            y_validation,
            predictions,
            pos_label=0,
            zero_division=0,
        )

        failure_f1 = f1_score(
            y_validation,
            predictions,
            pos_label=0,
            zero_division=0,
        )

        accuracy = accuracy_score(
            y_validation,
            predictions,
        )

        predicted_failures = (
            predictions == 0
        ).sum()

        print(
            f"{threshold:>20.3f}"
            f"{failure_precision:>20.4f}"
            f"{failure_recall:>18.4f}"
            f"{failure_f1:>15.4f}"
            f"{accuracy:>12.4f}"
            f"{predicted_failures:>16,}"
        )

    # --------------------------------------------------------
    # Full precision-recall curve
    # --------------------------------------------------------

    precision, recall, thresholds = (
        precision_recall_curve(
            y_validation,
            failure_probabilities,
            pos_label=0,
        )
    )

    print("\n" + "=" * 80)
    print("BEST VALIDATION THRESHOLD")
    print("=" * 80)

    # Maximum F1 for FAILURE class
    f1_scores = (
        2 * precision[:-1] * recall[:-1]
        / (
            precision[:-1]
            + recall[:-1]
            + 1e-12
        )
    )

    best_f1_index = np.argmax(
        f1_scores
    )

    best_f1_threshold = (
        thresholds[best_f1_index]
    )

    print(
        f"\nBest Failure F1:"
        f" {f1_scores[best_f1_index]:.4f}"
    )

    print(
        f"Threshold:"
        f" {best_f1_threshold:.6f}"
    )

    print(
        f"Failure Precision:"
        f" {precision[best_f1_index]:.4f}"
    )

    print(
        f"Failure Recall:"
        f" {recall[best_f1_index]:.4f}"
    )

    # --------------------------------------------------------
    # Target recall analysis
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("THRESHOLD FOR TARGET FAILURE RECALL")
    print("=" * 80)

    # target_recalls = [
    #     0.30,
    #     0.40,
    #     0.50,
    #     0.60,
    #     0.70,
    #     0.80,
    # ]

    # for target in target_recalls:

    #     valid_indices = np.where(
    #         recall[:-1] >= target
    #     )[0]

    #     if len(valid_indices) == 0:

    #         print(
    #             f"\nRecall >= {target:.0%}: "
    #             f"NOT ACHIEVABLE"
    #         )

    #         continue

    #     # Highest failure-probability threshold
    #     # that still achieves the desired recall.
    #     index = valid_indices[-1]

    #     print(
    #         f"\nRecall >= {target:.0%}"
    #     )

    #     print(
    #         f"  Threshold: "
    #         f"{thresholds[index]:.6f}"
    #     )

    #     print(
    #         f"  Precision:  "
    #         f"{precision[index]:.4f}"
    #     )

    #     print(
    #         f"  Recall:     "
    #         f"{recall[index]:.4f}"
    #     )

    # return best_f1_threshold

    eligible_indices = np.where(
        recall[:-1] >= TARGET_FAILURE_RECALL
    )[0]

    if len(eligible_indices) == 0:
        raise ValueError(
            f"No validation threshold achieves "
            f"{TARGET_FAILURE_RECALL:.0%} failure recall."
        )

    # Among thresholds that meet the recall goal, select the
    # highest-precision one. On a tie, use the higher threshold.
    eligible = pd.DataFrame({
        "threshold": thresholds[eligible_indices],
        "failure_precision": precision[:-1][eligible_indices],
        "failure_recall": recall[:-1][eligible_indices],
    })

    selected = eligible.sort_values(
        ["failure_precision", "threshold"],
        ascending=[False, False],
    ).iloc[0]

    selected_threshold = float(selected["threshold"])

    print(
        f"\nSelected for target failure recall "
        f"({TARGET_FAILURE_RECALL:.0%})"
    )
    print(f"Threshold: {selected_threshold:.6f}")
    print(f"Failure Precision: {selected['failure_precision']:.4f}")
    print(f"Failure Recall: {selected['failure_recall']:.4f}")

    return selected_threshold


# ============================================================
# HYPERPARAMETER SEARCH
# ============================================================

def tune_model(
    X_train,
    y_train,
    X_validation,
    y_validation,
    validation_df,
):

    preprocessor = build_preprocessor()

    print("\nFitting preprocessor once...")

    X_train_transformed = preprocessor.fit_transform(X_train)
    X_validation_transformed = preprocessor.transform(X_validation)

    print(f"Training matrix shape: {X_train_transformed.shape}")
    print(f"Validation matrix shape: {X_validation_transformed.shape}")

    parameter_sets = list(
        ParameterSampler(
            PARAMETER_SPACE,
            n_iter=N_SEARCH_ITERATIONS,
            random_state=RANDOM_STATE,
        )
    )

    best_cost = float("inf")
    best_model = None
    best_params = None

    failure_count = (y_train == 0).sum()
    success_count = (y_train == 1).sum()
    base_failure_weight = success_count / failure_count

    val_amount = validation_df["amount"].to_numpy()
    val_route_fee_pct = validation_df["route_cost_percent"].to_numpy() / 100.0

    print("\n" + "=" * 60)
    print("CLASS WEIGHTING")
    print("=" * 60)
    print(f"Failure count: {failure_count:,}")
    print(f"Success count: {success_count:,}")
    print(f"Base failure weight: {base_failure_weight:.4f}")
    print("(multiplier on this base weight is now searched per candidate)")

    print("\nStarting hyperparameter search (selecting by validation cost)...")
    print(f"Candidates: {len(parameter_sets)}")

    for i, raw_params in enumerate(parameter_sets, start=1):

        params = dict(raw_params)
        multiplier = params.pop("class_weight_multiplier")

        class_weight = {
            0: base_failure_weight * multiplier,
            1: 1.0,
        }

        print(f"\n[{i}/{len(parameter_sets)}]")
        print(f"class_weight_multiplier={multiplier}, params={params}")

        model, cost = train_candidate(
            params,
            X_train_transformed,
            y_train,
            X_validation_transformed,
            y_validation,
            class_weight,
            val_amount,
            val_route_fee_pct,
        )

        print(f"Best achievable validation cost/txn: {cost:.4f}")

        if cost < best_cost:
            best_cost = cost
            best_model = model
            best_params = raw_params
            print("NEW BEST MODEL")

    print("\n" + "=" * 60)
    print("BEST PARAMETERS")
    print("=" * 60)

    for key, value in best_params.items():  # type: ignore
        print(f"{key}: {value}")

    print(f"\nBest validation cost/txn: {best_cost:.6f}")

    return (best_model, preprocessor, best_params)


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(
    model,
    preprocessor,
    test_df,
    failure_threshold=0.5,
):

    X_test, y_test = prepare_xy(
        test_df
    )

    X_test_transformed = (
        preprocessor.transform(
            X_test
        )
    )

    # P(success)
    success_probabilities = (
        model.predict_proba(
            X_test_transformed
        )[:, 1]
    )

    # P(failure)
    failure_probabilities = (
        1.0 - success_probabilities
    )

    # Predict failure when P(failure)
    # >= selected threshold.
    predicted_failure = (
        failure_probabilities
        >= failure_threshold
    )

    # Convert:
    #
    # failure -> 0
    # success -> 1
    predictions = (
        ~predicted_failure
    ).astype(int)

    print("\n" + "=" * 60)
    print("FINAL TEST RESULTS")
    print("=" * 60)

    print(
        f"Failure Threshold: "
        f"{failure_threshold:.6f}"
    )

    print(
        f"\nROC-AUC:  "
        f"{roc_auc_score(y_test, success_probabilities):.6f}"
    )

    print(
        f"PR-AUC:   "
        f"{average_precision_score(y_test, success_probabilities):.6f}"
    )

    print(
        f"Log Loss: "
        f"{log_loss(y_test, success_probabilities):.6f}"
    )

    print(
        f"Brier:    "
        f"{brier_score_loss(y_test, success_probabilities):.6f}"
    )

    print(
        f"Accuracy: "
        f"{accuracy_score(y_test, predictions):.6f}"
    )

    print(
        f"Failure Precision: "
        f"{precision_score(y_test, predictions, pos_label=0, zero_division=0):.6f}"
    )

    print(
        f"Failure Recall: "
        f"{recall_score(y_test, predictions, pos_label=0, zero_division=0):.6f}"
    )

    print(
        f"Failure F1: "
        f"{f1_score(y_test, predictions, pos_label=0, zero_division=0):.6f}"
    )

    print("\nConfusion Matrix:")

    print(
        confusion_matrix(
            y_test,
            predictions,
        )
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
        )
    )

# ============================================================
# FEATURE IMPORTANCE ANALYSIS
# ============================================================

def analyze_feature_importance(
    model,
    preprocessor,
):

    print("\n" + "=" * 80)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("=" * 80)

    # --------------------------------------------------------
    # Get transformed feature names
    # --------------------------------------------------------

    feature_names = (
        preprocessor
        .get_feature_names_out()
    )

    importances = (
        model.feature_importances_
    )

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    })

    # --------------------------------------------------------
    # Remove pipeline prefixes
    #
    # numeric__amount
    # categorical__bank_BankA
    # --------------------------------------------------------

    importance_df["original_feature"] = (
        importance_df["feature"]
        .str.replace(
            "numeric__",
            "",
            regex=False,
        )
        .str.replace(
            "categorical__",
            "",
            regex=False,
        )
    )

    # --------------------------------------------------------
    # Aggregate one-hot features
    # --------------------------------------------------------

    aggregated = (
        importance_df
        .groupby(
            "original_feature",
            as_index=False,
        )["importance"]
        .sum()
        .sort_values(
            "importance",
            ascending=False,
        )
    ) # type: ignore

    # --------------------------------------------------------
    # Normalize to percentage
    # --------------------------------------------------------

    total_importance = (
        aggregated["importance"].sum()
    )

    aggregated["importance_pct"] = (
        aggregated["importance"]
        / total_importance
        * 100
    )

    print(
        f"\n{'Feature':<40}"
        f"{'Importance':>15}"
        f"{'Percentage':>15}"
    )

    print("-" * 70)

    for _, row in aggregated.iterrows():

        print(
            f"{row['original_feature']:<40}"
            f"{row['importance']:>15,.0f}"
            f"{row['importance_pct']:>14.2f}%"
        )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    importance_path = (
        "./models/feature_importance_v2.csv"
    )

    aggregated.to_csv(
        importance_path,
        index=False,
    )

    print(
        f"\nFeature importance saved to: "
        f"{importance_path}"
    )

    return aggregated


# ============================================================
# SAVE
# ============================================================

def save_model(
    model,
    preprocessor,
    best_params,
    threshold,
    path,
):

    directory = os.path.dirname(path)

    if directory:
        os.makedirs(
            directory,
            exist_ok=True,
        )

    artifact = {

        "model": model,

        "preprocessor": preprocessor,

        "threshold": threshold,

        "best_params": best_params,

        "numeric_features":
            NUMERIC_FEATURES,

        "categorical_features":
            CATEGORICAL_FEATURES,
    }

    joblib.dump(
        artifact,
        path,
    )

    print(
        f"\nModel saved to: {path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("Loading dataset...")

    (
        train_df,
        validation_df,
        test_df,
    ) = load_data()

    print(
        f"Training rows:   "
        f"{len(train_df):,}"
    )

    print(
        f"Validation rows: "
        f"{len(validation_df):,}"
    )

    print(
        f"Test rows:       "
        f"{len(test_df):,}"
    )

    # --------------------------------------------------------
    # Prepare train / validation data
    # --------------------------------------------------------

    X_train, y_train = (
        prepare_xy(train_df)
    )

    X_validation, y_validation = (
        prepare_xy(validation_df)
    )

    # --------------------------------------------------------
    # Hyperparameter tuning
    #
    # Validation is used ONLY for:
    # - selecting hyperparameters
    # - early stopping
    #
    # Test set remains untouched.
    # --------------------------------------------------------

    (
        model,
        preprocessor,
        best_params,
    ) = tune_model(
        X_train,
        y_train,
        X_validation,
        y_validation,
        validation_df,
    )

    feature_importance = (
        analyze_feature_importance(
            model,
            preprocessor,
        )
    )

    # --------------------------------------------------------
    # Threshold selection
    #
    # Still using validation only.
    # --------------------------------------------------------

    best_threshold = analyze_thresholds(
        model,
        preprocessor,
        validation_df,
    )

    print(
        f"\nSelected validation threshold: "
        f"{best_threshold:.6f}"
    )

    # --------------------------------------------------------
    # FINAL TEST EVALUATION
    #
    # At this point:
    #
    # - model is selected
    # - hyperparameters are selected
    # - early stopping is finished
    # - threshold is selected
    #
    # NOW we are allowed to touch test data.
    # --------------------------------------------------------

    evaluate_model(
        model,
        preprocessor,
        test_df,
        failure_threshold=best_threshold,
    )

    # --------------------------------------------------------
    # SAVE FINAL MODEL
    # --------------------------------------------------------

    save_model(
        model,
        preprocessor,
        best_params,
        best_threshold,
        MODEL_PATH,
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()