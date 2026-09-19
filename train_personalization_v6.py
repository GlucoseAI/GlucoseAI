import os
import json
import math
import joblib
import requests
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ============================================================
# CONFIG
# ============================================================

DATASET_FILE = "personalization_dataset.csv"

MODEL_FILE = "personalization_model_v6.joblib"
METRICS_FILE = "personalization_metrics_v6.json"

SUPABASE_URL = "https://tpzwxutiveixprniptyh.supabase.co"
SUPABASE_API_KEY = os.getenv("SUPABASE_API_KEY")

EXPERIMENTS_TABLE = "personalization_experiments"

MIN_TRAIN_RUNS = 3

ALPHA_VALUES = [
    0.1,
    0.3,
    1.0,
    3.0,
    10.0,
    30.0,
    100.0,
]

HORIZONS = list(range(5, 121, 5))

BASE_FEATURE_COLUMNS = [
    "current_glucose",
    "delta_5m",
    "delta_15m",
    "delta_30m",
    "delta_60m",
    "velocity_5m",
    "velocity_15m",
    "velocity_30m",
    "velocity_60m",
    "acceleration_5_15",
    "acceleration_15_30",
    "timesfm_prediction",
    "hour_sin",
    "hour_cos",
]

TARGET_COLUMN = "correction_target"


# ============================================================
# HELPERS
# ============================================================

def rmse(y_true, y_pred):
    return math.sqrt(mean_squared_error(y_true, y_pred))


def make_feature_matrix(df):
    """
    Create the same feature representation for every model.

    Base numerical features + one-hot horizon features.
    """

    X = df[BASE_FEATURE_COLUMNS].copy()

    for horizon in HORIZONS:
        X[f"horizon_{horizon}"] = (
            df["horizon_minutes"].astype(int) == horizon
        ).astype(float)

    return X


def create_model(alpha):
    """
    Ridge regression with standardized features.
    """

    return Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=alpha))
    ])


def calculate_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    errors = y_pred - y_true

    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(rmse(y_true, y_pred)),
        "bias": float(np.mean(errors)),
    }


def calculate_bias_model(train_df):
    """
    Mean correction calculated ONLY from training data.
    """

    return float(train_df[TARGET_COLUMN].mean())


def apply_bias_model(df, bias):
    return df["timesfm_prediction"].to_numpy() + bias


def safe_float(value):
    if value is None:
        return None

    if isinstance(value, (np.floating, np.integer)):
        return float(value)

    if isinstance(value, float):
        if not np.isfinite(value):
            return None

    return value


# ============================================================
# LOAD DATA
# ============================================================

print("Loading personalization dataset...")

df = pd.read_csv(DATASET_FILE)

print(f"Loaded {len(df)} rows.")

required_columns = (
    BASE_FEATURE_COLUMNS
    + [
        "forecast_created_at",
        "horizon_minutes",
        "actual_glucose",
        "timesfm_prediction",
        TARGET_COLUMN,
    ]
)

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise RuntimeError(
        "Missing required columns:\n"
        + "\n".join(missing_columns)
    )


# ============================================================
# CLEANING
# ============================================================

print(f"Rows before cleaning: {len(df)}")

df["forecast_created_at"] = pd.to_datetime(
    df["forecast_created_at"],
    utc=True,
    errors="coerce",
)

numeric_columns = (
    BASE_FEATURE_COLUMNS
    + [
        "horizon_minutes",
        "actual_glucose",
        TARGET_COLUMN,
    ]
)

for column in numeric_columns:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce",
    )

df = df.dropna(
    subset=required_columns
).copy()

df = df.sort_values(
    "forecast_created_at"
).reset_index(drop=True)

print(f"Rows after cleaning:  {len(df)}")


# ============================================================
# GROUP INTO FORECAST RUNS
# ============================================================

run_times = (
    df["forecast_created_at"]
    .drop_duplicates()
    .sort_values()
    .tolist()
)

print()
print("==========================================================")
print("DATASET INFORMATION")
print("==========================================================")
print(f"Rows:          {len(df)}")
print(f"Forecast runs: {len(run_times)}")
print()


if len(run_times) <= MIN_TRAIN_RUNS:
    raise RuntimeError(
        f"Not enough forecast runs. "
        f"Need more than {MIN_TRAIN_RUNS}, "
        f"have {len(run_times)}."
    )


# ============================================================
# ALPHA SEARCH
# ============================================================

print("==========================================================")
print("RIDGE ALPHA SEARCH")
print("==========================================================")
print()
print("Candidate alpha values:")
print(ALPHA_VALUES)
print()


# ============================================================
# INNER WALK-FORWARD ALPHA SELECTION
# ============================================================

def select_alpha_inner(train_df):
    """
    Select alpha using ONLY data available before the outer
    test run.

    This prevents the outer test data from influencing alpha.

    For a given outer training set:

        run 1 + run 2 -> validate run 3
        run 1 + run 2 + run 3 -> validate run 4
        ...

    If there is not enough data for an inner validation,
    alpha=1.0 is used as a conservative fallback.
    """

    inner_runs = (
        train_df["forecast_created_at"]
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    if len(inner_runs) < 4:
        return 1.0

    alpha_scores = {}

    for alpha in ALPHA_VALUES:

        validation_true = []
        validation_pred = []

        # Start with at least 3 training runs.
        for inner_index in range(3, len(inner_runs)):

            inner_train_runs = inner_runs[:inner_index]
            inner_validation_run = inner_runs[inner_index]

            inner_train = train_df[
                train_df["forecast_created_at"].isin(
                    inner_train_runs
                )
            ]

            inner_validation = train_df[
                train_df["forecast_created_at"]
                == inner_validation_run
            ]

            if len(inner_train) == 0:
                continue

            if len(inner_validation) == 0:
                continue

            model = create_model(alpha)

            X_train = make_feature_matrix(inner_train)
            y_train = inner_train[TARGET_COLUMN].to_numpy()

            X_val = make_feature_matrix(inner_validation)

            model.fit(
                X_train,
                y_train,
            )

            prediction = model.predict(X_val)

            validation_true.extend(
                inner_validation[TARGET_COLUMN].to_numpy()
            )

            validation_pred.extend(prediction)

        if len(validation_true) == 0:
            alpha_scores[alpha] = float("inf")
        else:
            alpha_scores[alpha] = mean_absolute_error(
                validation_true,
                validation_pred,
            )

    best_alpha = min(
        alpha_scores,
        key=alpha_scores.get
    )

    return float(best_alpha)


# ============================================================
# OUTER WALK-FORWARD BACKTEST
# ============================================================

print("==========================================================")
print("NESTED WALK-FORWARD BACKTEST - V6")
print("==========================================================")

outer_results = []

all_predictions = {
    alpha: {
        "y_true": [],
        "y_pred": [],
    }
    for alpha in ALPHA_VALUES
}

bias_predictions = []
bias_actuals = []

timesfm_predictions = []
timesfm_actuals = []

selected_alpha_history = []


for test_index in range(MIN_TRAIN_RUNS, len(run_times)):

    train_runs = run_times[:test_index]
    test_run = run_times[test_index]

    train_df = df[
        df["forecast_created_at"].isin(train_runs)
    ].copy()

    test_df = df[
        df["forecast_created_at"] == test_run
    ].copy()

    if len(train_df) == 0 or len(test_df) == 0:
        continue

    # --------------------------------------------------------
    # Select alpha using ONLY previous training runs.
    # --------------------------------------------------------

    selected_alpha = select_alpha_inner(train_df)

    selected_alpha_history.append({
        "test_run": test_run.isoformat(),
        "selected_alpha": selected_alpha,
    })

    # --------------------------------------------------------
    # TimesFM baseline
    # --------------------------------------------------------

    timesfm_actuals.extend(
        test_df["actual_glucose"].to_numpy()
    )

    timesfm_predictions.extend(
        test_df["timesfm_prediction"].to_numpy()
    )

    # --------------------------------------------------------
    # Mean bias correction
    # --------------------------------------------------------

    train_bias = calculate_bias_model(train_df)

    bias_pred = apply_bias_model(
        test_df,
        train_bias
    )

    bias_actuals.extend(
        test_df[TARGET_COLUMN].to_numpy()
    )

    bias_predictions.extend(
        bias_pred - test_df["timesfm_prediction"].to_numpy()
        + test_df[TARGET_COLUMN].to_numpy()
    )

    # The expression above reconstructs corrected glucose:
    # TimesFM prediction + learned correction.
    #
    # Since TARGET = actual - TimesFM,
    # this is:
    #
    # TimesFM + train_bias

    bias_predictions[-len(test_df):] = (
        test_df["timesfm_prediction"].to_numpy()
        + train_bias
    ).tolist()

    # --------------------------------------------------------
    # Test every alpha independently
    # --------------------------------------------------------

    for alpha in ALPHA_VALUES:

        model = create_model(alpha)

        X_train = make_feature_matrix(train_df)
        y_train = train_df[TARGET_COLUMN].to_numpy()

        X_test = make_feature_matrix(test_df)

        model.fit(
            X_train,
            y_train,
        )

        correction = model.predict(X_test)

        prediction = (
            test_df["timesfm_prediction"].to_numpy()
            + correction
        )

        all_predictions[alpha]["y_true"].extend(
            test_df["actual_glucose"].to_numpy()
        )

        all_predictions[alpha]["y_pred"].extend(
            prediction
        )

    # --------------------------------------------------------
    # Selected alpha model
    # --------------------------------------------------------

    selected_model = create_model(selected_alpha)

    X_train = make_feature_matrix(train_df)
    y_train = train_df[TARGET_COLUMN].to_numpy()

    X_test = make_feature_matrix(test_df)

    selected_model.fit(
        X_train,
        y_train,
    )

    selected_correction = selected_model.predict(
        X_test
    )

    selected_prediction = (
        test_df["timesfm_prediction"].to_numpy()
        + selected_correction
    )

    # --------------------------------------------------------
    # Run metrics
    # --------------------------------------------------------

    run_timesfm = calculate_metrics(
        test_df["actual_glucose"],
        test_df["timesfm_prediction"],
    )

    run_selected = calculate_metrics(
        test_df["actual_glucose"],
        selected_prediction,
    )

    print(
        f"Test run {test_index + 1}/{len(run_times)} | "
        f"{test_run} | "
        f"n={len(test_df)} | "
        f"alpha={selected_alpha} | "
        f"TimesFM MAE={run_timesfm['mae']:.3f} | "
        f"V6 MAE={run_selected['mae']:.3f}"
    )

    outer_results.append({
        "test_run": test_run.isoformat(),
        "samples": int(len(test_df)),
        "selected_alpha": selected_alpha,
        "timesfm_mae": run_timesfm["mae"],
        "timesfm_rmse": run_timesfm["rmse"],
        "timesfm_bias": run_timesfm["bias"],
        "v6_mae": run_selected["mae"],
        "v6_rmse": run_selected["rmse"],
        "v6_bias": run_selected["bias"],
    })


# ============================================================
# GLOBAL METRICS
# ============================================================

print()
print("==========================================================")
print("GLOBAL RESULTS - V6")
print("==========================================================")

timesfm_metrics = calculate_metrics(
    timesfm_actuals,
    timesfm_predictions,
)

bias_metrics = calculate_metrics(
    bias_actuals,
    bias_predictions,
)

alpha_metrics = {}

for alpha in ALPHA_VALUES:

    metrics = calculate_metrics(
        all_predictions[alpha]["y_true"],
        all_predictions[alpha]["y_pred"],
    )

    alpha_metrics[str(alpha)] = metrics


# Selected alpha OOF results

selected_actuals = []
selected_predictions = []

for result in outer_results:

    test_run = result["test_run"]

    test_df = df[
        df["forecast_created_at"].astype(str)
        == test_run
    ]

    if len(test_df) == 0:
        continue

    # We cannot reconstruct selected predictions here,
    # so use the stored per-run metrics for summary below.


# ============================================================
# OOF SELECTED MODEL METRICS
# ============================================================

# Re-run selected outer predictions so that we have the
# complete prediction vector for MAE/RMSE/bias.

selected_actuals = []
selected_predictions = []

for test_index in range(MIN_TRAIN_RUNS, len(run_times)):

    train_runs = run_times[:test_index]
    test_run = run_times[test_index]

    train_df = df[
        df["forecast_created_at"].isin(train_runs)
    ].copy()

    test_df = df[
        df["forecast_created_at"] == test_run
    ].copy()

    if len(train_df) == 0 or len(test_df) == 0:
        continue

    selected_alpha = select_alpha_inner(train_df)

    model = create_model(selected_alpha)

    X_train = make_feature_matrix(train_df)
    y_train = train_df[TARGET_COLUMN].to_numpy()

    X_test = make_feature_matrix(test_df)

    model.fit(
        X_train,
        y_train,
    )

    correction = model.predict(X_test)

    prediction = (
        test_df["timesfm_prediction"].to_numpy()
        + correction
    )

    selected_actuals.extend(
        test_df["actual_glucose"].to_numpy()
    )

    selected_predictions.extend(
        prediction
    )


v6_metrics = calculate_metrics(
    selected_actuals,
    selected_predictions,
)


# ============================================================
# PRINT ALPHA COMPARISON
# ============================================================

print()
print("==========================================================")
print("ALPHA COMPARISON")
print("==========================================================")

print()
print(
    f"{'Alpha':>10} | "
    f"{'MAE':>10} | "
    f"{'RMSE':>10} | "
    f"{'Bias':>10}"
)

print("-" * 50)

for alpha in ALPHA_VALUES:

    metrics = alpha_metrics[str(alpha)]

    print(
        f"{alpha:>10} | "
        f"{metrics['mae']:>10.3f} | "
        f"{metrics['rmse']:>10.3f} | "
        f"{metrics['bias']:>+10.3f}"
    )


# Best alpha based on OOF diagnostic.
#
# IMPORTANT:
# This is useful for comparison, but the selected V6 model
# uses nested walk-forward alpha selection above.

best_diagnostic_alpha = min(
    ALPHA_VALUES,
    key=lambda a: alpha_metrics[str(a)]["mae"]
)

best_diagnostic_metrics = alpha_metrics[
    str(best_diagnostic_alpha)
]


# ============================================================
# FINAL RESULTS
# ============================================================

print()
print("==========================================================")
print("WALK-FORWARD RESULTS - V6")
print("==========================================================")

print()
print("TIMESFM BASELINE")
print("----------------")
print(
    f"MAE  : {timesfm_metrics['mae']:.3f} mmol/l"
)
print(
    f"RMSE : {timesfm_metrics['rmse']:.3f} mmol/l"
)
print(
    f"Bias : {timesfm_metrics['bias']:+.3f} mmol/l"
)

print()
print("TIMESFM + MEAN BIAS CORRECTION")
print("-------------------------------")
print(
    f"MAE  : {bias_metrics['mae']:.3f} mmol/l"
)
print(
    f"RMSE : {bias_metrics['rmse']:.3f} mmol/l"
)
print(
    f"Bias : {bias_metrics['bias']:+.3f} mmol/l"
)

print()
print("BEST FIXED RIDGE ALPHA - DIAGNOSTIC")
print("------------------------------------")
print(
    f"Alpha: {best_diagnostic_alpha}"
)
print(
    f"MAE  : {best_diagnostic_metrics['mae']:.3f} mmol/l"
)
print(
    f"RMSE : {best_diagnostic_metrics['rmse']:.3f} mmol/l"
)
print(
    f"Bias : {best_diagnostic_metrics['bias']:+.3f} mmol/l"
)

print()
print("V6 NESTED WALK-FORWARD")
print("-----------------------")
print(
    f"MAE  : {v6_metrics['mae']:.3f} mmol/l"
)
print(
    f"RMSE : {v6_metrics['rmse']:.3f} mmol/l"
)
print(
    f"Bias : {v6_metrics['bias']:+.3f} mmol/l"
)


# ============================================================
# IMPROVEMENT
# ============================================================

improvement_vs_timesfm = (
    1
    - v6_metrics["mae"]
    / timesfm_metrics["mae"]
) * 100

improvement_vs_bias = (
    1
    - v6_metrics["mae"]
    / bias_metrics["mae"]
) * 100


print()
print("IMPROVEMENT")
print("-----------")

print(
    f"V6 vs TimesFM: "
    f"{improvement_vs_timesfm:+.1f}%"
)

print(
    f"V6 vs Bias correction: "
    f"{improvement_vs_bias:+.1f}%"
)


# ============================================================
# SELECTED ALPHA HISTORY
# ============================================================

print()
print("==========================================================")
print("SELECTED ALPHA BY OUTER TEST RUN")
print("==========================================================")

alpha_counts = {}

for item in selected_alpha_history:

    alpha = str(item["selected_alpha"])

    alpha_counts[alpha] = (
        alpha_counts.get(alpha, 0) + 1
    )

    print(
        f"{item['test_run']} -> "
        f"alpha={item['selected_alpha']}"
    )


# ============================================================
# TRAIN FINAL V6 MODEL
# ============================================================

print()
print("==========================================================")
print("TRAINING FINAL V6 MODEL")
print("==========================================================")

# Select final alpha using walk-forward over all currently
# available data, without using future data in each validation.

final_alpha = select_alpha_inner(df)

print(
    f"Final selected alpha: {final_alpha}"
)

final_model = create_model(final_alpha)

X_all = make_feature_matrix(df)
y_all = df[TARGET_COLUMN].to_numpy()

final_model.fit(
    X_all,
    y_all,
)


# ============================================================
# SAVE MODEL
# ============================================================

model_package = {
    "model": final_model,
    "model_name": "ridge_v6_nested_alpha",
    "alpha": final_alpha,
    "alpha_candidates": ALPHA_VALUES,
    "feature_columns": BASE_FEATURE_COLUMNS,
    "horizons": HORIZONS,
    "target": TARGET_COLUMN,
    "dataset_rows": int(len(df)),
    "forecast_runs": int(len(run_times)),
}

joblib.dump(
    model_package,
    MODEL_FILE,
)

print()
print(
    f"Final V6 model saved to {MODEL_FILE}"
)


# ============================================================
# SAVE METRICS JSON
# ============================================================

metrics_output = {
    "model_name": "ridge_v6_nested_alpha",
    "dataset_rows": int(len(df)),
    "forecast_runs": int(len(run_times)),
    "test_samples": int(len(selected_actuals)),
    "test_forecast_runs": int(len(outer_results)),

    "timesfm": timesfm_metrics,

    "bias_correction": bias_metrics,

    "v6_nested": v6_metrics,

    "best_diagnostic_alpha": float(
        best_diagnostic_alpha
    ),

    "best_diagnostic_alpha_metrics":
        best_diagnostic_metrics,

    "final_alpha": float(final_alpha),

    "alpha_results": alpha_metrics,

    "selected_alpha_history":
        selected_alpha_history,

    "outer_results":
        outer_results,
}


with open(
    METRICS_FILE,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        metrics_output,
        f,
        indent=2,
        ensure_ascii=False,
    )


print(
    f"Metrics saved to {METRICS_FILE}"
)


# ============================================================
# SAVE EXPERIMENT TO SUPABASE
# ============================================================

if SUPABASE_API_KEY:

    git_sha = os.getenv(
        "GITHUB_SHA",
        None,
    )

    experiment_payload = {
        "model_name": "ridge_v6_nested_alpha",
        "git_sha": git_sha,

        "dataset_rows": int(len(df)),
        "test_samples": int(len(selected_actuals)),
        "test_forecast_runs": int(len(outer_results)),

        "timesfm_mae":
            safe_float(timesfm_metrics["mae"]),

        "timesfm_rmse":
            safe_float(timesfm_metrics["rmse"]),

        "timesfm_bias":
            safe_float(timesfm_metrics["bias"]),

        "bias_mae":
            safe_float(bias_metrics["mae"]),

        "bias_rmse":
            safe_float(bias_metrics["rmse"]),

        "bias_bias":
            safe_float(bias_metrics["bias"]),

        "ridge_mae":
            safe_float(v6_metrics["mae"]),

        "ridge_rmse":
            safe_float(v6_metrics["rmse"]),

        "ridge_bias":
            safe_float(v6_metrics["bias"]),

        "horizon_results": {
            "model": "ridge_v6_nested_alpha",

            "final_alpha":
                float(final_alpha),

            "best_diagnostic_alpha":
                float(best_diagnostic_alpha),

            "alpha_results":
                alpha_metrics,

            "selected_alpha_counts":
                alpha_counts,

            "selected_alpha_history":
                selected_alpha_history,

            "outer_results":
                outer_results,
        },
    }

    headers = {
        "apikey": SUPABASE_API_KEY,
        "Authorization":
            f"Bearer {SUPABASE_API_KEY}",
        "Content-Type":
            "application/json",
        "Prefer":
            "return=minimal",
    }

    response = requests.post(
        f"{SUPABASE_URL}/rest/v1/"
        f"{EXPERIMENTS_TABLE}",
        headers=headers,
        json=experiment_payload,
        timeout=30,
    )

    if response.status_code not in [200, 201]:

        print(
            "WARNING: Could not save experiment "
            "to Supabase."
        )

        print(
            f"Status: {response.status_code}"
        )

        print(
            response.text
        )

    else:

        print(
            "Experiment saved to Supabase."
        )

else:

    print(
        "WARNING: SUPABASE_API_KEY is not available. "
        "Experiment was not saved to Supabase."
    )


# ============================================================
# FINISHED
# ============================================================

print()
print("==========================================================")
print("Personalization V6 completed successfully.")
print("==========================================================")
