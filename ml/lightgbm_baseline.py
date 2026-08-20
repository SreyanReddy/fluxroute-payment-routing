import os
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from lightgbm import LGBMClassifier

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    log_loss,
    brier_score_loss,
    confusion_matrix,
    classification_report,
)


# ============================================================
# CONFIG
# ============================================================

DATA_PATH = "./dataset-generator-routing/payment_dataset_features_4week.csv"
MODEL_PATH = "models/lightgbm_baseline.joblib"

TARGET = "success"

DROP_COLUMNS = [
    "txn_id",
    "dataset_split",
]


# ============================================================
# FEATURES
# ============================================================

NUMERIC_FEATURES = [
    "minute",
    "day_in_week",
    "hour_of_day",
    "amount",

    "route_base_success_rate",
    "route_base_latency_ms",
    "route_cost_percent",

    "route_success_rate_5m",
    "route_success_rate_15m",
    "route_avg_latency_5m",

    "route_current_load",
    "route_requests_1m",

    "time_since_last_failure",
    "route_flagged_down",

    "bank_route_success_rate_15m",
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
# LOAD DATA
# ============================================================

def load_data(path):
    df = pd.read_csv(path)

    train_df = df[df["dataset_split"] == "train"].copy()
    test_df = df[df["dataset_split"] == "test"].copy()

    return train_df, test_df


# ============================================================
# PREPROCESSING
# ============================================================

def build_preprocessor():

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=True,
        )),
    ])

    return ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
    ])


# ============================================================
# MODEL
# ============================================================

def build_model():

    preprocessor = build_preprocessor()

    classifier = LGBMClassifier(
        objective="binary",

        n_estimators=500,
        learning_rate=0.05,

        num_leaves=31,
        max_depth=-1,

        subsample=0.8,
        colsample_bytree=0.8,

        reg_alpha=0.1,
        reg_lambda=0.1,

        random_state=42,
        n_jobs=-1,

        verbosity=-1,
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", classifier),
    ])


# ============================================================
# TRAIN
# ============================================================

def train_model(model, train_df):

    X_train = train_df.drop(
        columns=[TARGET, *DROP_COLUMNS]
    )

    y_train = train_df[TARGET]

    model.fit(
        X_train,
        y_train,
    )

    return model


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(model, test_df):

    X_test = test_df.drop(
        columns=[TARGET, *DROP_COLUMNS]
    )

    y_test = test_df[TARGET]

    probabilities = model.predict_proba(X_test)[:, 1]

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    print("\n" + "=" * 60)
    print("LIGHTGBM EVALUATION")
    print("=" * 60)

    print(
        f"ROC-AUC:  "
        f"{roc_auc_score(y_test, probabilities):.6f}"
    )

    print(
        f"PR-AUC:   "
        f"{average_precision_score(y_test, probabilities):.6f}"
    )

    print(
        f"Log Loss: "
        f"{log_loss(y_test, probabilities):.6f}"
    )

    print(
        f"Brier:    "
        f"{brier_score_loss(y_test, probabilities):.6f}"
    )

    print(
        f"Accuracy: "
        f"{accuracy_score(y_test, predictions):.6f}"
    )

    print("\nConfusion Matrix:")

    print(
        confusion_matrix(
            y_test,
            predictions
        )
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions
        )
    )


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def show_feature_importance(model, top_n=30):

    preprocessor = model.named_steps[
        "preprocessor"
    ]

    classifier = model.named_steps[
        "model"
    ]

    feature_names = (
        preprocessor
        .get_feature_names_out()
    )

    importance = classifier.feature_importances_

    result = pd.DataFrame({
        "feature": feature_names,
        "importance": importance,
    })

    result = result.sort_values(
        "importance",
        ascending=False,
    )

    print("\n" + "=" * 60)
    print(f"TOP {top_n} FEATURES")
    print("=" * 60)

    print(
        result.head(top_n)
        .to_string(index=False)
    )


# ============================================================
# SAVE
# ============================================================

def save_model(model, path):

    directory = os.path.dirname(path)

    if directory:
        os.makedirs(
            directory,
            exist_ok=True
        )

    joblib.dump(
        model,
        path
    )

    print(
        f"\nModel saved to: {path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("Loading dataset...")

    train_df, test_df = load_data(
        DATA_PATH
    )

    print(
        f"Train rows: "
        f"{len(train_df):,}"
    )

    print(
        f"Test rows:  "
        f"{len(test_df):,}"
    )

    print("\nBuilding LightGBM model...")

    model = build_model()

    print("Training...")

    model = train_model(
        model,
        train_df
    )

    print("Training complete.")

    evaluate_model(
        model,
        test_df
    )

    show_feature_importance(
        model,
        top_n=30
    )

    save_model(
        model,
        MODEL_PATH
    )


if __name__ == "__main__":
    main()