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

MODEL_FILE = "personalization_model.joblib"

METRICS_FILE = "personalization_metrics.json"

RIDGE_ALPHA = 1.0

MIN_TRAIN_RUNS = 3


SUPABASE_URL = (
    "https://tpzwxutiveixprniptyh.supabase.co"
)

EXPERIMENT_TABLE = (
    "personalization_experiments"
)


# =========================================================
# BASE FEATURES
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
# HORIZONS
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

    if not os.path.exists(
        DATASET_FILE
    ):

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
    # Numeric columns
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

            df["horizon_minutes"]
            ==
            horizon

        ).astype(float)

    return df


# =========================================================
# CREATE MODEL
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

        .reset_index(drop=True)

    )


# =========================================================
# CALCULATE METRICS
# =========================================================

def calculate_metrics(
    actual,
    predicted
):

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
        "WALK-FORWARD BACKTEST"
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


    # -----------------------------------------------------
    # Each future forecast run becomes a test
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
        # HORIZON FEATURES
        # -------------------------------------------------

        train_df = add_horizon_features(
            train_df
        )

        test_df = add_horizon_features(
            test_df
        )


        # -------------------------------------------------
        # TRAIN RIDGE
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
        # TEST
        # -------------------------------------------------

        X_test = test_df[
            FEATURE_COLUMNS
        ]


        actual = test_df[
            "actual_glucose"
        ].to_numpy()


        timesfm = test_df[
            "timesfm_prediction"
        ].to_numpy()


        # -------------------------------------------------
        # TIMESFM
        # -------------------------------------------------

        timesfm_metrics = calculate_metrics(

            actual,
            timesfm

        )


        # -------------------------------------------------
        # MEAN BIAS CORRECTION
        # -----------------------------------------------------

        train_bias = np.mean(

            train_df[
                TARGET_COLUMN
            ].to_numpy()

        )


        bias_corrected = (

            timesfm +
            train_bias

        )


        bias_metrics = calculate_metrics(

            actual,
            bias_corrected

        )


        # -------------------------------------------------
        # RIDGE
        # -------------------------------------------------

        predicted_correction = model.predict(

            X_test

        )


        personalized = (

            timesfm +
            predicted_correction

        )


        personalized_metrics = calculate_metrics(

            actual,
            personalized

        )


        # -------------------------------------------------
        # SAVE INDIVIDUAL RESULTS
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
                        test_df.iloc[i][
                            "horizon_minutes"
                        ]
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

                "personalized_prediction":

                    float(
                        personalized[i]
                    ),

                "timesfm_error":

                    float(
                        actual[i] -
                        timesfm[i]
                    ),

                "bias_error":

                    float(
                        actual[i] -
                        bias_corrected[i]
                    ),

                "personalized_error":

                    float(
                        actual[i] -
                        personalized[i]
                    ),

            })


        # -------------------------------------------------
        # PRINT CURRENT RUN
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

            f"Ridge MAE="
            f"{personalized_metrics['mae']:.3f}"

        )


    return pd.DataFrame(
        all_results
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


    personalized = results[
        "personalized_prediction"
    ].to_numpy()


    timesfm_metrics = calculate_metrics(

        actual,
        timesfm

    )


    bias_metrics = calculate_metrics(

        actual,
        bias

    )


    personalized_metrics = calculate_metrics(

        actual,
        personalized

    )


    return {

        "samples":
            int(len(results)),

        "forecast_runs":
            int(
                results[
                    "forecast_created_at"
                ]
                .nunique()
            ),

        "timesfm":
            timesfm_metrics,

        "bias_correction":
            bias_metrics,

        "personalized":
            personalized_metrics,

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


        personalized = group[
            "personalized_prediction"
        ].to_numpy()


        timesfm_metrics = calculate_metrics(

            actual,
            timesfm

        )


        bias_metrics = calculate_metrics(

            actual,
            bias

        )


        personalized_metrics = calculate_metrics(

            actual,
            personalized

        )


        rows.append({

            "horizon_minutes":
                int(horizon),

            "samples":
                len(group),

            "timesfm_mae":
                timesfm_metrics["mae"],

            "bias_mae":
                bias_metrics["mae"],

            "personalized_mae":
                personalized_metrics["mae"],

            "timesfm_rmse":
                timesfm_metrics["rmse"],

            "bias_rmse":
                bias_metrics["rmse"],

            "personalized_rmse":
                personalized_metrics["rmse"],

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
        "WALK-FORWARD RESULTS"
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
    # TIMESFM
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
    # BIAS
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
    # RIDGE
    # -----------------------------------------------------

    print()

    print(
        "TIMESFM + RIDGE PERSONALIZATION"
    )

    print(
        "--------------------------------"
    )


    print(

        f"MAE  : "
        f"{metrics['personalized']['mae']:.3f} mmol/l"

    )


    print(

        f"RMSE : "
        f"{metrics['personalized']['rmse']:.3f} mmol/l"

    )


    print(

        f"Bias : "
        f"{metrics['personalized']['bias']:+.3f} mmol/l"

    )


    # -----------------------------------------------------
    # HORIZON
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
        "TimesFM | Bias | Ridge"
    )


    print(
        "--------+---------+---------+------+------"
    )


    for _, row in horizon_results.iterrows():

        print(

            f"{int(row['horizon_minutes']):7d} | "

            f"{int(row['samples']):7d} | "

            f"{row['timesfm_mae']:7.3f} | "

            f"{row['bias_mae']:4.3f} | "

            f"{row['personalized_mae']:5.3f}"

        )


# =========================================================
# SAVE METRICS
# =========================================================

def save_metrics(
    metrics,
    horizon_results
):

    output = {

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
        f"Metrics saved to {METRICS_FILE}"
    )


# =========================================================
# SAVE EXPERIMENT TO SUPABASE
# =========================================================

def save_experiment_to_supabase(
    metrics,
    horizon_results,
    dataset_rows
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


    github_sha = os.environ.get(
        "GITHUB_SHA"
    )


    payload = {

        "model_name":
            "ridge_v4",

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
                metrics["personalized"]["mae"]
            ),

        "ridge_rmse":
            float(
                metrics["personalized"]["rmse"]
            ),

        "ridge_bias":
            float(
                metrics["personalized"]["bias"]
            ),

        "horizon_results":
            horizon_json,

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
# TRAIN FINAL MODEL
# =========================================================

def train_final_model(
    df
):

    print()

    print(
        "Training final Ridge model "
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


    joblib.dump(

        model,

        MODEL_FILE

    )


    print(
        f"Final model saved to "
        f"{MODEL_FILE}"
    )


# =========================================================
# MAIN
# =========================================================

def main():

    # -----------------------------------------------------
    # LOAD
    # -----------------------------------------------------

    df = load_dataset()


    # -----------------------------------------------------
    # PREPARE
    # -----------------------------------------------------

    df = prepare_data(
        df
    )


    # -----------------------------------------------------
    # WALK-FORWARD
    # -----------------------------------------------------

    results = walk_forward_backtest(
        df
    )


    if results.empty:

        raise RuntimeError(

            "Walk-forward backtest "
            "produced no results."

        )


    # -----------------------------------------------------
    # METRICS
    # -----------------------------------------------------

    metrics = calculate_overall_results(
        results
    )


    horizon_results = calculate_horizon_results(
        results
    )


    # -----------------------------------------------------
    # PRINT
    # -----------------------------------------------------

    print_results(

        metrics,

        horizon_results

    )


    # -----------------------------------------------------
    # SAVE LOCAL METRICS
    # -----------------------------------------------------

    save_metrics(

        metrics,

        horizon_results

    )


    # -----------------------------------------------------
    # SAVE EXPERIMENT TO SUPABASE
    # -----------------------------------------------------

    save_experiment_to_supabase(

        metrics,

        horizon_results,

        len(df)

    )


    # -----------------------------------------------------
    # TRAIN FINAL MODEL
    # -----------------------------------------------------

    train_final_model(
        df
    )


    print()

    print(
        "=========================================================="
    )

    print(
        "Personalization v4 completed successfully."
    )

    print(
        "=========================================================="
    )


# =========================================================
# START
# =========================================================

if __name__ == "__main__":

    main()
