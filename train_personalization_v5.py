import os
import json
import joblib
import requests

import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error


# =========================================================
# SETTINGS
# =========================================================

DATASET_FILE = "personalization_dataset.csv"

MODEL_FILE = "personalization_model_v5.joblib"

METRICS_FILE = "personalization_metrics_v5.json"

RIDGE_ALPHA = 1.0

MIN_TRAIN_RUNS = 3

# Minimum number of historical samples needed
# to learn a separate weight for a horizon.
MIN_HORIZON_SAMPLES = 4

# Shrink horizon-specific weights toward
# the global weight to reduce overfitting.
HORIZON_WEIGHT_SHRINKAGE = 3.0

# Allowed correction weight range.
MIN_CORRECTION_WEIGHT = 0.0
MAX_CORRECTION_WEIGHT = 1.5


SUPABASE_URL = (
    "https://tpzwxutiveixprniptyh.supabase.co"
)

EXPERIMENT_TABLE = (
    "personalization_experiments"
)


# =========================================================
# FEATURES
# =========================================================

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


# =========================================================
# FORECAST HORIZONS
# =========================================================

HORIZONS = list(
    range(5, 125, 5)
)


HORIZON_FEATURE_COLUMNS = [

    f"horizon_{h}"

    for h in HORIZONS

]


FEATURE_COLUMNS = (

    BASE_FEATURE_COLUMNS

    +

    HORIZON_FEATURE_COLUMNS

)


# =========================================================
# LOAD DATASET
# =========================================================

def load_dataset():

    print(
        "Loading personalization dataset..."
    )

    if not os.path.exists(DATASET_FILE):

        raise FileNotFoundError(

            f"Dataset not found: "
            f"{DATASET_FILE}"

        )

    df = pd.read_csv(
        DATASET_FILE
    )

    print(
        f"Loaded {len(df)} rows."
    )

    return df


# =========================================================
# PREPARE DATA
# =========================================================

def prepare_data(df):

    required_columns = (

        BASE_FEATURE_COLUMNS

        +

        [
            "horizon_minutes",
            TARGET_COLUMN,
            "forecast_created_at",
            "actual_glucose",
        ]

    )


    missing = [

        column

        for column in required_columns

        if column not in df.columns

    ]


    if missing:

        raise RuntimeError(

            "Dataset is missing columns: "

            +

            ", ".join(missing)

        )


    # -----------------------------------------------------
    # Datetime
    # -----------------------------------------------------

    df[
        "forecast_created_at"
    ] = pd.to_datetime(

        df[
            "forecast_created_at"
        ],

        utc=True

    )


    # -----------------------------------------------------
    # Numeric conversion
    # -----------------------------------------------------

    numeric_columns = (

        BASE_FEATURE_COLUMNS

        +

        [
            "horizon_minutes",
            TARGET_COLUMN,
            "actual_glucose",
        ]

    )


    for column in numeric_columns:

        df[column] = pd.to_numeric(

            df[column],

            errors="coerce"

        )


    # -----------------------------------------------------
    # Remove incomplete rows
    # -----------------------------------------------------

    before = len(df)


    df = df.dropna(

        subset=(

            BASE_FEATURE_COLUMNS

            +

            [
                "horizon_minutes",
                TARGET_COLUMN,
                "actual_glucose",
            ]

        )

    )


    # -----------------------------------------------------
    # Sort chronologically
    # -----------------------------------------------------

    df = (

        df

        .sort_values(
            "forecast_created_at"
        )

        .reset_index(
            drop=True
        )

    )


    after = len(df)


    print(
        f"Rows before cleaning: {before}"
    )

    print(
        f"Rows after cleaning:  {after}"
    )


    return df


# =========================================================
# ADD HORIZON FEATURES
# =========================================================

def add_horizon_features(df):

    df = df.copy()


    for horizon in HORIZONS:

        column = (
            f"horizon_{horizon}"
        )


        df[column] = (

            df[
                "horizon_minutes"
            ]

            ==

            horizon

        ).astype(float)


    return df


# =========================================================
# CREATE RIDGE MODEL
# =========================================================

def create_model():

    return Pipeline(

        [

            (
                "scaler",

                StandardScaler()

            ),

            (
                "ridge",

                Ridge(
                    alpha=RIDGE_ALPHA
                )

            ),

        ]

    )


# =========================================================
# GET FORECAST RUNS
# =========================================================

def get_forecast_runs(df):

    return (

        df[
            "forecast_created_at"
        ]

        .drop_duplicates()

        .sort_values()

        .reset_index(
            drop=True
        )

    )


# =========================================================
# CALCULATE METRICS
# =========================================================

def calculate_metrics(
    actual,
    predicted
):

    actual = np.asarray(
        actual,
        dtype=float
    )

    predicted = np.asarray(
        predicted,
        dtype=float
    )


    errors = (
        actual -
        predicted
    )


    mae = mean_absolute_error(

        actual,

        predicted

    )


    rmse = np.sqrt(

        mean_squared_error(

            actual,

            predicted

        )

    )


    bias = np.mean(
        errors
    )


    return {

        "mae":
            float(mae),

        "rmse":
            float(rmse),

        "bias":
            float(bias),

    }


# =========================================================
# LEARN GLOBAL CORRECTION WEIGHT
# =========================================================

def learn_global_weight(
    correction,
    target
):

    correction = np.asarray(
        correction,
        dtype=float
    )

    target = np.asarray(
        target,
        dtype=float
    )


    denominator = (

        np.sum(
            correction ** 2
        )

        +

        1e-8

    )


    weight = (

        np.sum(
            correction * target
        )

        /

        denominator

    )


    weight = np.clip(

        weight,

        MIN_CORRECTION_WEIGHT,

        MAX_CORRECTION_WEIGHT

    )


    return float(weight)


# =========================================================
# LEARN HORIZON WEIGHTS
# =========================================================

def learn_horizon_weights(
    train_df,
    predicted_correction
):

    train_df = train_df.copy()


    target = train_df[
        TARGET_COLUMN
    ].to_numpy(
        dtype=float
    )


    correction = np.asarray(
        predicted_correction,
        dtype=float
    )


    # -----------------------------------------------------
    # Global correction weight
    # -----------------------------------------------------

    global_weight = learn_global_weight(

        correction,

        target

    )


    weights = {}


    # -----------------------------------------------------
    # Individual horizon weights
    # -----------------------------------------------------

    for horizon in HORIZONS:

        mask = (

            train_df[
                "horizon_minutes"
            ].to_numpy()

            ==

            horizon

        )


        count = int(
            np.sum(mask)
        )


        # -------------------------------------------------
        # No historical data for this horizon
        # -------------------------------------------------

        if count == 0:

            weights[horizon] = {

                "weight":
                    global_weight,

                "samples":
                    0,

                "source":
                    "global"

            }

            continue


        horizon_correction = (
            correction[mask]
        )

        horizon_target = (
            target[mask]
        )


        # -------------------------------------------------
        # Too little data:
        # use global weight.
        # -------------------------------------------------

        if count < MIN_HORIZON_SAMPLES:

            weights[horizon] = {

                "weight":
                    global_weight,

                "samples":
                    count,

                "source":
                    "global"

            }

            continue


        # -------------------------------------------------
        # Raw horizon-specific weight
        # -------------------------------------------------

        raw_weight = learn_global_weight(

            horizon_correction,

            horizon_target

        )


        # -------------------------------------------------
        # Shrink toward global weight.
        #
        # This prevents extreme weights when there are
        # only a few samples for a specific horizon.
        # -------------------------------------------------

        shrunk_weight = (

            (

                count * raw_weight

                +

                HORIZON_WEIGHT_SHRINKAGE
                * global_weight

            )

            /

            (

                count
                +
                HORIZON_WEIGHT_SHRINKAGE

            )

        )


        shrunk_weight = np.clip(

            shrunk_weight,

            MIN_CORRECTION_WEIGHT,

            MAX_CORRECTION_WEIGHT

        )


        weights[horizon] = {

            "weight":
                float(shrunk_weight),

            "raw_weight":
                float(raw_weight),

            "samples":
                count,

            "source":
                "horizon"

        }


    return {

        "global_weight":
            float(global_weight),

        "horizon_weights":
            weights

    }


# =========================================================
# APPLY HORIZON WEIGHTS
# =========================================================

def apply_horizon_weights(
    timesfm,
    predicted_correction,
    horizon_values,
    weighting
):

    timesfm = np.asarray(
        timesfm,
        dtype=float
    )

    predicted_correction = np.asarray(
        predicted_correction,
        dtype=float
    )

    horizon_values = np.asarray(
        horizon_values,
        dtype=int
    )


    global_weight = weighting[
        "global_weight"
    ]


    horizon_weights = weighting[
        "horizon_weights"
    ]


    final_prediction = np.empty(
        len(timesfm),
        dtype=float
    )


    used_weights = []


    for i in range(
        len(timesfm)
    ):

        horizon = int(
            horizon_values[i]
        )


        info = horizon_weights.get(
            horizon
        )


        if info is None:

            weight = global_weight

        else:

            weight = info[
                "weight"
            ]


        final_prediction[i] = (

            timesfm[i]

            +

            weight
            * predicted_correction[i]

        )


        used_weights.append(
            weight
        )


    return (

        final_prediction,

        np.asarray(
            used_weights,
            dtype=float
        )

    )


# =========================================================
# WALK-FORWARD BACKTEST
# =========================================================

def walk_forward_backtest(df):

    runs = get_forecast_runs(
        df
    )


    print()

    print(
        "=========================================================="
    )

    print(
        "WALK-FORWARD BACKTEST - V5"
    )

    print(
        "=========================================================="
    )

    print(
        f"Forecast runs available: "
        f"{len(runs)}"
    )


    if len(runs) <= MIN_TRAIN_RUNS:

        raise RuntimeError(

            "Not enough independent forecast runs "
            "for walk-forward backtesting."

        )


    all_results = []

    all_weight_results = []


    # -----------------------------------------------------
    # Walk forward through forecast runs.
    # -----------------------------------------------------

    for test_index in range(

        MIN_TRAIN_RUNS,

        len(runs)

    ):

        train_runs = set(

            runs.iloc[
                :test_index
            ]

        )


        test_run = runs.iloc[
            test_index
        ]


        train_df = df[

            df[
                "forecast_created_at"
            ].isin(train_runs)

        ].copy()


        test_df = df[

            df[
                "forecast_created_at"
            ]

            ==

            test_run

        ].copy()


        if train_df.empty:

            continue


        if test_df.empty:

            continue


        # -------------------------------------------------
        # Horizon one-hot features
        # -------------------------------------------------

        train_df = add_horizon_features(
            train_df
        )

        test_df = add_horizon_features(
            test_df
        )


        # -------------------------------------------------
        # Train Ridge
        # -------------------------------------------------

        model = create_model()


        X_train = train_df[
            FEATURE_COLUMNS
        ]

        y_train = train_df[
            TARGET_COLUMN
        ]


        model.fit(

            X_train,

            y_train

        )


        # -------------------------------------------------
        # Test data
        # -------------------------------------------------

        X_test = test_df[
            FEATURE_COLUMNS
        ]


        actual = test_df[
            "actual_glucose"
        ].to_numpy(
            dtype=float
        )


        timesfm = test_df[
            "timesfm_prediction"
        ].to_numpy(
            dtype=float
        )


        horizons = test_df[
            "horizon_minutes"
        ].to_numpy(
            dtype=int
        )


        # -------------------------------------------------
        # 1. TimesFM baseline
        # -------------------------------------------------

        timesfm_metrics = calculate_metrics(

            actual,

            timesfm

        )


        # -------------------------------------------------
        # 2. Mean bias correction
        #
        # IMPORTANT:
        # bias is calculated only from training data.
        # -------------------------------------------------

        train_bias = np.mean(

            train_df[
                TARGET_COLUMN
            ].to_numpy(
                dtype=float
            )

        )


        bias_corrected = (

            timesfm

            +

            train_bias

        )


        bias_metrics = calculate_metrics(

            actual,

            bias_corrected

        )


        # -------------------------------------------------
        # 3. Ridge v4
        # -------------------------------------------------

        predicted_correction = model.predict(

            X_test

        )


        ridge_v4 = (

            timesfm

            +

            predicted_correction

        )


        ridge_v4_metrics = calculate_metrics(

            actual,

            ridge_v4

        )


        # -------------------------------------------------
        # 4. Learn horizon weights
        #
        # IMPORTANT:
        # weights are learned ONLY using train data.
        # -------------------------------------------------

        train_predicted_correction = model.predict(

            train_df[
                FEATURE_COLUMNS
            ]

        )


        weighting = learn_horizon_weights(

            train_df,

            train_predicted_correction

        )


        # -------------------------------------------------
        # Apply horizon-specific weighting
        # -------------------------------------------------

        ridge_v5, used_weights = (
            apply_horizon_weights(

                timesfm,

                predicted_correction,

                horizons,

                weighting

            )
        )


        ridge_v5_metrics = calculate_metrics(

            actual,

            ridge_v5

        )


        # -------------------------------------------------
        # Store horizon weights
        # -------------------------------------------------

        for horizon in HORIZONS:

            info = weighting[
                "horizon_weights"
            ][horizon]


            all_weight_results.append({

                "forecast_created_at":
                    test_run,

                "horizon_minutes":
                    horizon,

                "weight":
                    float(
                        info["weight"]
                    ),

                "samples":
                    int(
                        info["samples"]
                    ),

                "source":
                    info["source"],

            })


        # -------------------------------------------------
        # Store individual predictions
        # -------------------------------------------------

        for i in range(
            len(test_df)
        ):

            all_results.append({

                "forecast_created_at":

                    test_df.iloc[i][
                        "forecast_created_at"
                    ],


                "horizon_minutes":

                    int(
                        horizons[i]
                    ),


                "actual_glucose":

                    float(
                        actual[i]
                    ),


                "timesfm_prediction":

                    float(
                        timesfm[i]
                    ),


                "bias_prediction":

                    float(
                        bias_corrected[i]
                    ),


                "ridge_v4_prediction":

                    float(
                        ridge_v4[i]
                    ),


                "ridge_v5_prediction":

                    float(
                        ridge_v5[i]
                    ),


                "timesfm_error":

                    float(
                        actual[i]
                        -
                        timesfm[i]
                    ),


                "bias_error":

                    float(
                        actual[i]
                        -
                        bias_corrected[i]
                    ),


                "ridge_v4_error":

                    float(
                        actual[i]
                        -
                        ridge_v4[i]
                    ),


                "ridge_v5_error":

                    float(
                        actual[i]
                        -
                        ridge_v5[i]
                    ),


                "ridge_v5_weight":

                    float(
                        used_weights[i]
                    ),

            })


        # -------------------------------------------------
        # Print run result
        # -------------------------------------------------

        print(

            f"Test run "
            f"{test_index + 1}/"
            f"{len(runs)} | "

            f"{test_run} | "

            f"n={len(test_df)} | "

            f"TimesFM MAE="
            f"{timesfm_metrics['mae']:.3f} | "

            f"Bias MAE="
            f"{bias_metrics['mae']:.3f} | "

            f"Ridge v4 MAE="
            f"{ridge_v4_metrics['mae']:.3f} | "

            f"Ridge v5 MAE="
            f"{ridge_v5_metrics['mae']:.3f}"

        )


    return (

        pd.DataFrame(
            all_results
        ),

        pd.DataFrame(
            all_weight_results
        )

    )


# =========================================================
# OVERALL RESULTS
# =========================================================

def calculate_overall_results(
    results
):

    actual = results[
        "actual_glucose"
    ].to_numpy()


    timesfm = results[
        "timesfm_prediction"
    ].to_numpy()


    bias = results[
        "bias_prediction"
    ].to_numpy()


    ridge_v4 = results[
        "ridge_v4_prediction"
    ].to_numpy()


    ridge_v5 = results[
        "ridge_v5_prediction"
    ].to_numpy()


    return {

        "samples":
            int(
                len(results)
            ),


        "forecast_runs":
            int(

                results[
                    "forecast_created_at"
                ]

                .nunique()

            ),


        "timesfm":
            calculate_metrics(
                actual,
                timesfm
            ),


        "bias_correction":
            calculate_metrics(
                actual,
                bias
            ),


        "ridge_v4":
            calculate_metrics(
                actual,
                ridge_v4
            ),


        "ridge_v5":
            calculate_metrics(
                actual,
                ridge_v5
            ),

    }


# =========================================================
# RESULTS BY HORIZON
# =========================================================

def calculate_horizon_results(
    results
):

    rows = []


    for horizon, group in (

        results.groupby(
            "horizon_minutes"
        )

    ):

        actual = group[
            "actual_glucose"
        ].to_numpy()


        timesfm = group[
            "timesfm_prediction"
        ].to_numpy()


        bias = group[
            "bias_prediction"
        ].to_numpy()


        ridge_v4 = group[
            "ridge_v4_prediction"
        ].to_numpy()


        ridge_v5 = group[
            "ridge_v5_prediction"
        ].to_numpy()


        rows.append({

            "horizon_minutes":
                int(horizon),

            "samples":
                int(len(group)),

            "timesfm_mae":
                calculate_metrics(
                    actual,
                    timesfm
                )["mae"],

            "bias_mae":
                calculate_metrics(
                    actual,
                    bias
                )["mae"],

            "ridge_v4_mae":
                calculate_metrics(
                    actual,
                    ridge_v4
                )["mae"],

            "ridge_v5_mae":
                calculate_metrics(
                    actual,
                    ridge_v5
                )["mae"],

            "timesfm_rmse":
                calculate_metrics(
                    actual,
                    timesfm
                )["rmse"],

            "bias_rmse":
                calculate_metrics(
                    actual,
                    bias
                )["rmse"],

            "ridge_v4_rmse":
                calculate_metrics(
                    actual,
                    ridge_v4
                )["rmse"],

            "ridge_v5_rmse":
                calculate_metrics(
                    actual,
                    ridge_v5
                )["rmse"],

        })


    return (

        pd.DataFrame(rows)

        .sort_values(
            "horizon_minutes"
        )

    )


# =========================================================
# PRINT RESULTS
# =========================================================

def print_results(
    metrics,
    horizon_results
):

    print()

    print(
        "=========================================================="
    )

    print(
        "WALK-FORWARD RESULTS - V5"
    )

    print(
        "=========================================================="
    )


    print()

    print(
        f"Test samples: "
        f"{metrics['samples']}"
    )

    print(
        f"Test forecast runs: "
        f"{metrics['forecast_runs']}"
    )


    # -----------------------------------------------------
    # TimesFM
    # -----------------------------------------------------

    print()

    print(
        "TIMESFM BASELINE"
    )

    print(
        "----------------"
    )

    print(

        f"MAE  : "
        f"{metrics['timesfm']['mae']:.3f} mmol/l"

    )

    print(

        f"RMSE : "
        f"{metrics['timesfm']['rmse']:.3f} mmol/l"

    )

    print(

        f"Bias : "
        f"{metrics['timesfm']['bias']:+.3f} mmol/l"

    )


    # -----------------------------------------------------
    # Bias correction
    # -----------------------------------------------------

    print()

    print(
        "TIMESFM + MEAN BIAS CORRECTION"
    )

    print(
        "-------------------------------"
    )

    print(

        f"MAE  : "
        f"{metrics['bias_correction']['mae']:.3f} mmol/l"

    )

    print(

        f"RMSE : "
        f"{metrics['bias_correction']['rmse']:.3f} mmol/l"

    )

    print(

        f"Bias : "
        f"{metrics['bias_correction']['bias']:+.3f} mmol/l"

    )


    # -----------------------------------------------------
    # Ridge v4
    # -----------------------------------------------------

    print()

    print(
        "TIMESFM + RIDGE V4"
    )

    print(
        "------------------"
    )

    print(

        f"MAE  : "
        f"{metrics['ridge_v4']['mae']:.3f} mmol/l"

    )

    print(

        f"RMSE : "
        f"{metrics['ridge_v4']['rmse']:.3f} mmol/l"

    )

    print(

        f"Bias : "
        f"{metrics['ridge_v4']['bias']:+.3f} mmol/l"

    )


    # -----------------------------------------------------
    # Ridge v5
    # -----------------------------------------------------

    print()

    print(
        "TIMESFM + RIDGE V5 HORIZON WEIGHTED"
    )

    print(
        "------------------------------------"
    )

    print(

        f"MAE  : "
        f"{metrics['ridge_v5']['mae']:.3f} mmol/l"

    )

    print(

        f"RMSE : "
        f"{metrics['ridge_v5']['rmse']:.3f} mmol/l"

    )

    print(

        f"Bias : "
        f"{metrics['ridge_v5']['bias']:+.3f} mmol/l"

    )


    # -----------------------------------------------------
    # Improvement
    # -----------------------------------------------------

    baseline_mae = (
        metrics["timesfm"]["mae"]
    )

    v4_mae = (
        metrics["ridge_v4"]["mae"]
    )

    v5_mae = (
        metrics["ridge_v5"]["mae"]
    )


    v4_improvement = (

        100.0

        *

        (
            baseline_mae
            -
            v4_mae
        )

        /

        baseline_mae

    )


    v5_improvement = (

        100.0

        *

        (
            baseline_mae
            -
            v5_mae
        )

        /

        baseline_mae

    )


    print()

    print(
        "IMPROVEMENT VS TIMESFM"
    )

    print(
        "----------------------"
    )

    print(

        f"Ridge v4: "
        f"{v4_improvement:+.1f}%"

    )

    print(

        f"Ridge v5: "
        f"{v5_improvement:+.1f}%"

    )


    # -----------------------------------------------------
    # Horizon results
    # -----------------------------------------------------

    print()

    print(
        "RESULTS BY HORIZON"
    )

    print(
        "------------------"
    )

    print()

    print(

        "Horizon | Samples | "
        "TimesFM | Bias | "
        "Ridge v4 | Ridge v5"

    )

    print(

        "--------+---------+---------+------+"
        "----------+---------"

    )


    for _, row in horizon_results.iterrows():

        print(

            f"{int(row['horizon_minutes']):7d} | "

            f"{int(row['samples']):7d} | "

            f"{row['timesfm_mae']:7.3f} | "

            f"{row['bias_mae']:4.3f} | "

            f"{row['ridge_v4_mae']:8.3f} | "

            f"{row['ridge_v5_mae']:7.3f}"

        )


# =========================================================
# SAVE METRICS
# =========================================================

def save_metrics(
    metrics,
    horizon_results
):

    output = {

        "model":
            "ridge_v5_horizon_weighted",


        "settings": {

            "ridge_alpha":
                RIDGE_ALPHA,

            "min_horizon_samples":
                MIN_HORIZON_SAMPLES,

            "horizon_weight_shrinkage":
                HORIZON_WEIGHT_SHRINKAGE,

            "min_correction_weight":
                MIN_CORRECTION_WEIGHT,

            "max_correction_weight":
                MAX_CORRECTION_WEIGHT,

        },


        "overall":
            metrics,


        "by_horizon":

            horizon_results

            .replace(
                {
                    np.nan: None
                }
            )

            .to_dict(
                orient="records"
            ),

    }


    with open(

        METRICS_FILE,

        "w",

        encoding="utf-8"

    ) as file:

        json.dump(

            output,

            file,

            indent=2

        )


    print()

    print(
        f"Metrics saved to "
        f"{METRICS_FILE}"
    )


# =========================================================
# SAVE EXPERIMENT TO SUPABASE
# =========================================================

def save_experiment_to_supabase(
    metrics,
    horizon_results,
    dataset_rows,
    weight_results
):

    api_key = os.environ.get(
        "SUPABASE_API_KEY"
    )


    if not api_key:

        print(
            "SUPABASE_API_KEY not available."
        )

        print(
            "Experiment will not be saved "
            "to Supabase."
        )

        return


    url = (

        f"{SUPABASE_URL}/rest/v1/"

        f"{EXPERIMENT_TABLE}"

    )


    # -----------------------------------------------------
    # Horizon metrics
    # -----------------------------------------------------

    horizon_json = (

        horizon_results

        .replace(
            {
                np.nan: None
            }
        )

        .to_dict(
            orient="records"
        )

    )


    # -----------------------------------------------------
    # Horizon weights
    # -----------------------------------------------------

    weight_json = (

        weight_results

        .replace(
            {
                np.nan: None
            }
        )

        .to_dict(
            orient="records"
        )

    )


    github_sha = os.environ.get(
        "GITHUB_SHA"
    )


    payload = {

        "model_name":
            "ridge_v5_horizon_weighted",


        "git_sha":
            github_sha,


        "dataset_rows":
            int(dataset_rows),


        "test_samples":
            int(
                metrics["samples"]
            ),


        "test_forecast_runs":
            int(
                metrics["forecast_runs"]
            ),


        "timesfm_mae":
            float(
                metrics["timesfm"]["mae"]
            ),


        "timesfm_rmse":
            float(
                metrics["timesfm"]["rmse"]
            ),


        "timesfm_bias":
            float(
                metrics["timesfm"]["bias"]
            ),


        "bias_mae":
            float(
                metrics["bias_correction"]["mae"]
            ),


        "bias_rmse":
            float(
                metrics["bias_correction"]["rmse"]
            ),


        "bias_bias":
            float(
                metrics["bias_correction"]["bias"]
            ),


        "ridge_mae":
            float(
                metrics["ridge_v5"]["mae"]
            ),


        "ridge_rmse":
            float(
                metrics["ridge_v5"]["rmse"]
            ),


        "ridge_bias":
            float(
                metrics["ridge_v5"]["bias"]
            ),


        "horizon_results": {

            "metrics":
                horizon_json,

            "weights":
                weight_json,

        },

    }


    headers = {

        "apikey":
            api_key,

        "Authorization":
            f"Bearer {api_key}",

        "Content-Type":
            "application/json",

        "Prefer":
            "return=minimal",

    }


    response = requests.post(

        url,

        headers=headers,

        json=payload,

        timeout=30,

    )


    if not response.ok:

        print(
            "Failed to save experiment:"
        )

        print(
            f"HTTP {response.status_code}"
        )

        print(
            response.text
        )

        response.raise_for_status()


    print()

    print(
        "Experiment saved to Supabase."
    )


# =========================================================
# TRAIN FINAL V5 MODEL
# =========================================================

def train_final_model(
    df
):

    print()

    print(
        "Training final Ridge v5 model "
        "on all available data..."
    )


    df = add_horizon_features(
        df
    )


    model = create_model()


    X = df[
        FEATURE_COLUMNS
    ]

    y = df[
        TARGET_COLUMN
    ]


    model.fit(

        X,

        y

    )


    # -----------------------------------------------------
    # Learn final horizon weights using all data.
    # -----------------------------------------------------

    train_predicted_correction = model.predict(
        X
    )


    weighting = learn_horizon_weights(

        df,

        train_predicted_correction

    )


    # -----------------------------------------------------
    # Store everything needed later for inference.
    # -----------------------------------------------------

    model_package = {

        "model":
            model,

        "global_weight":
            weighting[
                "global_weight"
            ],

        "horizon_weights":
            weighting[
                "horizon_weights"
            ],

        "feature_columns":
            FEATURE_COLUMNS,

        "ridge_alpha":
            RIDGE_ALPHA,

        "model_name":
            "ridge_v5_horizon_weighted",

    }


    joblib.dump(

        model_package,

        MODEL_FILE

    )


    print()

    print(

        f"Final v5 model saved to "
        f"{MODEL_FILE}"

    )


    # -----------------------------------------------------
    # Print final weights.
    # -----------------------------------------------------

    print()

    print(
        "FINAL HORIZON WEIGHTS"
    )

    print(
        "---------------------"
    )


    print(

        f"Global weight: "
        f"{weighting['global_weight']:.4f}"

    )


    for horizon in HORIZONS:

        info = weighting[
            "horizon_weights"
        ][horizon]


        print(

            f"{horizon:3d} min | "

            f"weight="
            f"{info['weight']:.4f} | "

            f"samples="
            f"{info['samples']:3d} | "

            f"{info['source']}"

        )


    return weighting


# =========================================================
# MAIN
# =========================================================

def main():

    # -----------------------------------------------------
    # Load dataset
    # -----------------------------------------------------

    df = load_dataset()


    # -----------------------------------------------------
    # Clean and prepare
    # -----------------------------------------------------

    df = prepare_data(
        df
    )


    # -----------------------------------------------------
    # Walk-forward backtest
    # -----------------------------------------------------

    (

        results,

        weight_results

    ) = walk_forward_backtest(
        df
    )


    if results.empty:

        raise RuntimeError(

            "Walk-forward backtest "
            "produced no results."

        )


    # -----------------------------------------------------
    # Calculate overall metrics
    # -----------------------------------------------------

    metrics = calculate_overall_results(
        results
    )


    # -----------------------------------------------------
    # Calculate horizon metrics
    # -----------------------------------------------------

    horizon_results = calculate_horizon_results(
        results
    )


    # -----------------------------------------------------
    # Print results
    # -----------------------------------------------------

    print_results(

        metrics,

        horizon_results

    )


    # -----------------------------------------------------
    # Save local metrics
    # -----------------------------------------------------

    save_metrics(

        metrics,

        horizon_results

    )


    # -----------------------------------------------------
    # Save experiment to Supabase
    # -----------------------------------------------------

    save_experiment_to_supabase(

        metrics,

        horizon_results,

        len(df),

        weight_results

    )


    # -----------------------------------------------------
    # Train final model
    # -----------------------------------------------------

    train_final_model(
        df
    )


    # -----------------------------------------------------
    # Done
    # -----------------------------------------------------

    print()

    print(
        "=========================================================="
    )

    print(
        "Personalization v5 completed successfully."
    )

    print(
        "=========================================================="
    )


# =========================================================
# START PROGRAM
# =========================================================

if __name__ == "__main__":

    main()
