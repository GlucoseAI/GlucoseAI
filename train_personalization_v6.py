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

MODEL_FILE = "personalization_model_v6_1.joblib"
METRICS_FILE = "personalization_metrics_v6_1.json"

SUPABASE_URL = "https://tpzwxutiveixprniptyh.supabase.co"
SUPABASE_API_KEY = os.getenv("SUPABASE_API_KEY")

EXPERIMENTS_TABLE = "personalization_experiments"

MODEL_NAME = "ridge_v6_1_alpha100"

# Alpha selected from the previous V6 experiment.
# It is now FIXED and will not be selected from future
# test data.
RIDGE_ALPHA = 100.0

MIN_TRAIN_RUNS = 3

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
    return math.sqrt(
        mean_squared_error(y_true, y_pred)
    )


def calculate_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    errors = y_pred - y_true

    return {
        "mae": float(
            mean_absolute_error(y_true, y_pred)
        ),
        "rmse": float(
            rmse(y_true, y_pred)
        ),
        "bias": float(
            np.mean(errors)
        ),
    }


def make_feature_matrix(df):
    """
    Create the feature matrix.

    Numerical glucose-history features +
    TimesFM prediction +
    time-of-day +
    one-hot horizon features.
    """

    X = df[BASE_FEATURE_COLUMNS].copy()

    for horizon in HORIZONS:
        X[f"horizon_{horizon}"] = (
            df["horizon_minutes"].astype(int)
            == horizon
        ).astype(float)

    return X


def create_model():
    """
    V6.1 model.

    Alpha is intentionally fixed at 100.
    """

    return Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "ridge",
            Ridge(alpha=RIDGE_ALPHA)
        ),
    ])


def calculate_training_bias(train_df):
    """
    Mean correction calculated ONLY from training data.

    correction_target =
        actual_glucose - timesfm_prediction
    """

    return float(
        train_df[TARGET_COLUMN].mean()
    )


def safe_float(value):
    if value is None:
        return None

    if isinstance(
        value,
        (np.floating, np.integer)
    ):
        return float(value)

    if isinstance(value, float):
        if not np.isfinite(value):
            return None

    return value


# ============================================================
# LOAD DATA
# ============================================================

print("Loading personalization dataset...")

df = pd.read_csv(
    DATASET_FILE
)

print(
    f"Loaded {len(df)} rows."
)


# ============================================================
# REQUIRED COLUMNS
# ============================================================

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
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:

    raise RuntimeError(
        "Missing required columns:\n"
        + "\n".join(missing_columns)
    )


# ============================================================
# CLEAN DATA
# ============================================================

print(
    f"Rows before cleaning: {len(df)}"
)

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

print(
    f"Rows after cleaning:  {len(df)}"
)


# ============================================================
# FORECAST RUNS
# ============================================================

run_times = (
    df["forecast_created_at"]
    .drop_duplicates()
    .sort_values()
    .tolist()
)

print()
print(
    "=========================================================="
)
print(
    "DATASET INFORMATION"
)
print(
    "=========================================================="
)

print(
    f"Rows:          {len(df)}"
)

print(
    f"Forecast runs: {len(run_times)}"
)

print(
    f"Fixed Ridge alpha: {RIDGE_ALPHA}"
)


if len(run_times) <= MIN_TRAIN_RUNS:

    raise RuntimeError(
        f"Not enough forecast runs. "
        f"Need more than {MIN_TRAIN_RUNS}, "
        f"have {len(run_times)}."
    )


# ============================================================
# WALK-FORWARD BACKTEST
# ============================================================

print()
print(
    "=========================================================="
)
print(
    "WALK-FORWARD BACKTEST - V6.1"
)
print(
    "=========================================================="
)

print()
print(
    "Ridge alpha is fixed at 100.0."
)

print(
    "No alpha selection is performed."
)

print()


# ------------------------------------------------------------
# Storage
# ------------------------------------------------------------

timesfm_actuals = []
timesfm_predictions = []

bias_actuals = []
bias_predictions = []

ridge_actuals = []
ridge_predictions = []

outer_results = []


# ============================================================
# OUTER WALK-FORWARD
# ============================================================

for test_index in range(
    MIN_TRAIN_RUNS,
    len(run_times)
):

    train_runs = run_times[
        :test_index
    ]

    test_run = run_times[
        test_index
    ]

    train_df = df[
        df["forecast_created_at"].isin(
            train_runs
        )
    ].copy()

    test_df = df[
        df["forecast_created_at"]
        == test_run
    ].copy()

    if (
        len(train_df) == 0
        or len(test_df) == 0
    ):
        continue

    # ========================================================
    # TIMESFM BASELINE
    # ========================================================

    actual = (
        test_df["actual_glucose"]
        .to_numpy()
    )

    timesfm = (
        test_df["timesfm_prediction"]
        .to_numpy()
    )

    timesfm_actuals.extend(
        actual
    )

    timesfm_predictions.extend(
        timesfm
    )

    # ========================================================
    # MEAN BIAS CORRECTION
    # ========================================================

    train_bias = calculate_training_bias(
        train_df
    )

    bias_prediction = (
        timesfm
        + train_bias
    )

    bias_actuals.extend(
        actual
    )

    bias_predictions.extend(
        bias_prediction
    )

    # ========================================================
    # RIDGE V6.1
    # ========================================================

    model = create_model()

    X_train = make_feature_matrix(
        train_df
    )

    y_train = (
        train_df[TARGET_COLUMN]
        .to_numpy()
    )

    X_test = make_feature_matrix(
        test_df
    )

    model.fit(
        X_train,
        y_train
    )

    correction = model.predict(
        X_test
    )

    ridge_prediction = (
        timesfm
        + correction
    )

    ridge_actuals.extend(
        actual
    )

    ridge_predictions.extend(
        ridge_prediction
    )

    # ========================================================
    # RUN METRICS
    # ========================================================

    run_timesfm = calculate_metrics(
        actual,
        timesfm
    )

    run_bias = calculate_metrics(
        actual,
        bias_prediction
    )

    run_ridge = calculate_metrics(
        actual,
        ridge_prediction
    )

    print(
        f"Test run "
        f"{test_index + 1}/{len(run_times)} | "
        f"{test_run} | "
        f"n={len(test_df)} | "
        f"TimesFM MAE="
        f"{run_timesfm['mae']:.3f} | "
        f"Bias MAE="
        f"{run_bias['mae']:.3f} | "
        f"V6.1 MAE="
        f"{run_ridge['mae']:.3f}"
    )

    outer_results.append({

        "test_run":
            test_run.isoformat(),

        "samples":
            int(len(test_df)),

        "ridge_alpha":
            RIDGE_ALPHA,

        "timesfm_mae":
            run_timesfm["mae"],

        "timesfm_rmse":
            run_timesfm["rmse"],

        "timesfm_bias":
            run_timesfm["bias"],

        "bias_mae":
            run_bias["mae"],

        "bias_rmse":
            run_bias["rmse"],

        "bias_bias":
            run_bias["bias"],

        "v6_1_mae":
            run_ridge["mae"],

        "v6_1_rmse":
            run_ridge["rmse"],

        "v6_1_bias":
            run_ridge["bias"],
    })


# ============================================================
# GLOBAL METRICS
# ============================================================

timesfm_metrics = calculate_metrics(
    timesfm_actuals,
    timesfm_predictions
)

bias_metrics = calculate_metrics(
    bias_actuals,
    bias_predictions
)

ridge_metrics = calculate_metrics(
    ridge_actuals,
    ridge_predictions
)


# ============================================================
# HORIZON METRICS
# ============================================================

print()
print(
    "=========================================================="
)
print(
    "RESULTS BY HORIZON"
)
print(
    "=========================================================="
)

print()
print(
    "Horizon | Samples | TimesFM | Bias | V6.1"
)
print(
    "--------+---------+---------+------+------"
)

horizon_results = {}

for horizon in HORIZONS:

    horizon_df = df[
        df["horizon_minutes"]
        == horizon
    ]

    if len(horizon_df) == 0:
        continue

    # Only calculate using rows that belong to
    # outer walk-forward test data.

    horizon_times = []
    horizon_actual = []
    horizon_ridge = []
    horizon_bias = []

    for result in outer_results:

        test_run = result["test_run"]

        test_run_timestamp = pd.Timestamp(
            test_run
        )

        test_rows = horizon_df[
            horizon_df[
                "forecast_created_at"
            ]
            == test_run_timestamp
        ]

        if len(test_rows) == 0:
            continue

        train_runs = [
            run
            for run in run_times
            if run < test_run_timestamp
        ]

        if len(train_runs) < MIN_TRAIN_RUNS:
            continue

        train_df = df[
            df["forecast_created_at"].isin(
                train_runs
            )
        ].copy()

        model = create_model()

        model.fit(
            make_feature_matrix(
                train_df
            ),
            train_df[
                TARGET_COLUMN
            ].to_numpy()
        )

        correction = model.predict(
            make_feature_matrix(
                test_rows
            )
        )

        actual_values = (
            test_rows[
                "actual_glucose"
            ].to_numpy()
        )

        timesfm_values = (
            test_rows[
                "timesfm_prediction"
            ].to_numpy()
        )

        train_bias = calculate_training_bias(
            train_df
        )

        bias_values = (
            timesfm_values
            + train_bias
        )

        ridge_values = (
            timesfm_values
            + correction
        )

        horizon_actual.extend(
            actual_values
        )

        horizon_times.extend(
            timesfm_values
        )

        horizon_bias.extend(
            bias_values
        )

        horizon_ridge.extend(
            ridge_values
        )

    if len(horizon_actual) == 0:
        continue

    h_timesfm = calculate_metrics(
        horizon_actual,
        horizon_times
    )

    h_bias = calculate_metrics(
        horizon_actual,
        horizon_bias
    )

    h_ridge = calculate_metrics(
        horizon_actual,
        horizon_ridge
    )

    horizon_results[
        str(horizon)
    ] = {

        "samples":
            int(len(horizon_actual)),

        "timesfm":
            h_timesfm,

        "bias":
            h_bias,

        "v6_1":
            h_ridge,
    }

    print(
        f"{horizon:7d} | "
        f"{len(horizon_actual):7d} | "
        f"{h_timesfm['mae']:7.3f} | "
        f"{h_bias['mae']:4.3f} | "
        f"{h_ridge['mae']:4.3f}"
    )


# ============================================================
# FINAL RESULTS
# ============================================================

print()
print(
    "=========================================================="
)
print(
    "WALK-FORWARD RESULTS - V6.1"
)
print(
    "=========================================================="
)

print()
print(
    "TIMESFM BASELINE"
)
print(
    "----------------"
)

print(
    f"MAE  : "
    f"{timesfm_metrics['mae']:.3f} mmol/l"
)

print(
    f"RMSE : "
    f"{timesfm_metrics['rmse']:.3f} mmol/l"
)

print(
    f"Bias : "
    f"{timesfm_metrics['bias']:+.3f} mmol/l"
)


print()
print(
    "TIMESFM + MEAN BIAS CORRECTION"
)
print(
    "-------------------------------"
)

print(
    f"MAE  : "
    f"{bias_metrics['mae']:.3f} mmol/l"
)

print(
    f"RMSE : "
    f"{bias_metrics['rmse']:.3f} mmol/l"
)

print(
    f"Bias : "
    f"{bias_metrics['bias']:+.3f} mmol/l"
)


print()
print(
    "TIMESFM + RIDGE V6.1"
)
print(
    "--------------------"
)

print(
    f"Alpha: "
    f"{RIDGE_ALPHA}"
)

print(
    f"MAE  : "
    f"{ridge_metrics['mae']:.3f} mmol/l"
)

print(
    f"RMSE : "
    f"{ridge_metrics['rmse']:.3f} mmol/l"
)

print(
    f"Bias : "
    f"{ridge_metrics['bias']:+.3f} mmol/l"
)


# ============================================================
# IMPROVEMENTS
# ============================================================

improvement_vs_timesfm = (
    1
    - ridge_metrics["mae"]
    / timesfm_metrics["mae"]
) * 100

improvement_vs_bias = (
    1
    - ridge_metrics["mae"]
    / bias_metrics["mae"]
) * 100


print()
print(
    "IMPROVEMENT"
)
print(
    "-----------"
)

print(
    f"V6.1 vs TimesFM: "
    f"{improvement_vs_timesfm:+.1f}%"
)

print(
    f"V6.1 vs Bias correction: "
    f"{improvement_vs_bias:+.1f}%"
)


# ============================================================
# TRAIN FINAL MODEL
# ============================================================

print()
print(
    "=========================================================="
)
print(
    "TRAINING FINAL V6.1 MODEL"
)
print(
    "=========================================================="
)

final_model = create_model()

X_all = make_feature_matrix(
    df
)

y_all = (
    df[TARGET_COLUMN]
    .to_numpy()
)

final_model.fit(
    X_all,
    y_all
)


# ============================================================
# SAVE MODEL
# ============================================================

model_package = {

    "model":
        final_model,

    "model_name":
        MODEL_NAME,

    "alpha":
        RIDGE_ALPHA,

    "feature_columns":
        BASE_FEATURE_COLUMNS,

    "horizons":
        HORIZONS,

    "target":
        TARGET_COLUMN,

    "dataset_rows":
        int(len(df)),

    "forecast_runs":
        int(len(run_times)),
}


joblib.dump(
    model_package,
    MODEL_FILE
)

print()
print(
    f"Final V6.1 model saved to "
    f"{MODEL_FILE}"
)


# ============================================================
# SAVE METRICS
# ============================================================

metrics_output = {

    "model_name":
        MODEL_NAME,

    "alpha":
        RIDGE_ALPHA,

    "dataset_rows":
        int(len(df)),

    "forecast_runs":
        int(len(run_times)),

    "test_samples":
        int(len(ridge_actuals)),

    "test_forecast_runs":
        int(len(outer_results)),

    "timesfm":
        timesfm_metrics,

    "bias_correction":
        bias_metrics,

    "v6_1":
        ridge_metrics,

    "improvement_vs_timesfm_percent":
        float(improvement_vs_timesfm),

    "improvement_vs_bias_percent":
        float(improvement_vs_bias),

    "horizon_results":
        horizon_results,

    "outer_results":
        outer_results,
}


with open(
    METRICS_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        metrics_output,
        f,
        indent=2,
        ensure_ascii=False
    )


print(
    f"Metrics saved to "
    f"{METRICS_FILE}"
)


# ============================================================
# SAVE EXPERIMENT TO SUPABASE
# ============================================================

if SUPABASE_API_KEY:

    git_sha = os.getenv(
        "GITHUB_SHA",
        None
    )

    experiment_payload = {

        "model_name":
            MODEL_NAME,

        "git_sha":
            git_sha,

        "dataset_rows":
            int(len(df)),

        "test_samples":
            int(len(ridge_actuals)),

        "test_forecast_runs":
            int(len(outer_results)),

        "timesfm_mae":
            safe_float(
                timesfm_metrics["mae"]
            ),

        "timesfm_rmse":
            safe_float(
                timesfm_metrics["rmse"]
            ),

        "timesfm_bias":
            safe_float(
                timesfm_metrics["bias"]
            ),

        "bias_mae":
            safe_float(
                bias_metrics["mae"]
            ),

        "bias_rmse":
            safe_float(
                bias_metrics["rmse"]
            ),

        "bias_bias":
            safe_float(
                bias_metrics["bias"]
            ),

        "ridge_mae":
            safe_float(
                ridge_metrics["mae"]
            ),

        "ridge_rmse":
            safe_float(
                ridge_metrics["rmse"]
            ),

        "ridge_bias":
            safe_float(
                ridge_metrics["bias"]
            ),

        "horizon_results": {
            "alpha":
                RIDGE_ALPHA,

            "horizon_metrics":
                horizon_results,

            "outer_results":
                outer_results,
        },
    }

    headers = {

        "apikey":
            SUPABASE_API_KEY,

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

    if response.status_code not in [
        200,
        201
    ]:

        print(
            "WARNING: Could not save "
            "experiment to Supabase."
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
        "WARNING: SUPABASE_API_KEY is not "
        "available. Experiment was not "
        "saved to Supabase."
    )


# ============================================================
# FINISHED
# ============================================================

print()
print(
    "=========================================================="
)
print(
    "Personalization V6.1 completed successfully."
)
print(
    "=========================================================="
)
