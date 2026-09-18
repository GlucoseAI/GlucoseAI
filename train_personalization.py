import os
import json
import joblib

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

# Minimum number of forecast runs required before we
# start testing the personalization model.
MIN_TRAIN_RUNS = 3


# =========================================================
# FEATURES
# =========================================================

FEATURE_COLUMNS = [

    "current_glucose",

    "delta_5m",

    "delta_15m",

    "delta_30m",

    "delta_60m",

    "timesfm_prediction",

    "horizon_minutes",

    "hour_sin",

    "hour_cos",

]


TARGET_COLUMN = "correction_target"


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

        FEATURE_COLUMNS
        +
        [
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

        FEATURE_COLUMNS
        +
        [
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

        subset=

            FEATURE_COLUMNS
            +
            [
                TARGET_COLUMN,
                "actual_glucose",
            ]

    )


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
# CREATE FORECAST RUNS
# =========================================================

def get_forecast_runs(df):

    runs = (

        df[
            "forecast_created_at"
        ]

        .drop_duplicates()

        .sort_values()

        .reset_index(drop=True)

    )


    return runs


# =========================================================
# CREATE MODEL
# =========================================================

def create_model():

    model = Pipeline(

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


    return model


# =========================================================
# WALK-FORWARD BACKTEST
# =========================================================

def walk_forward_backtest(df):

    runs = get_forecast_runs(df)


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
        f"Forecast runs available: {len(runs)}"
    )


    if len(runs) <= MIN_TRAIN_RUNS:

        raise RuntimeError(

            "Not enough independent forecast runs "
            "for walk-forward backtesting."

        )


    all_results = []


    # -----------------------------------------------------
    # Each future run becomes a separate test
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
            ] == test_run
        ].copy()


        if len(train_df) == 0:

            continue


        if len(test_df) == 0:

            continue


        # -------------------------------------------------
        # Train model only on past data
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
        # Test
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


        correction = model.predict(
            X_test
        )


        personalized = (

            timesfm +
            correction

        )


        # -------------------------------------------------
        # Save each prediction
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

                "personalized_prediction":
                    float(
                        personalized[i]
                    ),

                "timesfm_error":
                    float(
                        actual[i] -
                        timesfm[i]
                    ),

                "personalized_error":
                    float(
                        actual[i] -
                        personalized[i]
                    ),

            })


        # -------------------------------------------------
        # Print current test run
        # -------------------------------------------------

        timesfm_mae = mean_absolute_error(

            actual,

            timesfm

        )


        personalized_mae = mean_absolute_error(

            actual,

            personalized

        )


        print(

            f"Test run "
            f"{test_index + 1}/"
            f"{len(runs)} | "

            f"{test_run} | "

            f"n={len(test_df)} | "

            f"TimesFM MAE="
            f"{timesfm_mae:.3f} | "

            f"Personalized MAE="
            f"{personalized_mae:.3f}"

        )


    return pd.DataFrame(
        all_results
    )


# =========================================================
# CALCULATE OVERALL RESULTS
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


    personalized = results[
        "personalized_prediction"
    ].to_numpy()


    # -----------------------------------------------------
    # TimesFM
    # -----------------------------------------------------

    timesfm_error = (

        actual -
        timesfm

    )


    timesfm_mae = mean_absolute_error(

        actual,

        timesfm

    )


    timesfm_rmse = np.sqrt(

        mean_squared_error(

            actual,

            timesfm

        )

    )


    timesfm_bias = np.mean(
        timesfm_error
    )


    # -----------------------------------------------------
    # Personalization
    # -----------------------------------------------------

    personalized_error = (

        actual -
        personalized

    )


    personalized_mae = mean_absolute_error(

        actual,

        personalized

    )


    personalized_rmse = np.sqrt(

        mean_squared_error(

            actual,

            personalized

        )

    )


    personalized_bias = np.mean(
        personalized_error
    )


    # -----------------------------------------------------
    # Improvement
    # -----------------------------------------------------

    mae_improvement = (

        (
            timesfm_mae -
            personalized_mae
        )
        /
        timesfm_mae
        *
        100

    )


    rmse_improvement = (

        (
            timesfm_rmse -
            personalized_rmse
        )
        /
        timesfm_rmse
        *
        100

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

        "timesfm": {

            "mae":
                float(
                    timesfm_mae
                ),

            "rmse":
                float(
                    timesfm_rmse
                ),

            "bias":
                float(
                    timesfm_bias
                ),

        },

        "personalized": {

            "mae":
                float(
                    personalized_mae
                ),

            "rmse":
                float(
                    personalized_rmse
                ),

            "bias":
                float(
                    personalized_bias
                ),

        },

        "improvement": {

            "mae_percent":
                float(
                    mae_improvement
                ),

            "rmse_percent":
                float(
                    rmse_improvement
                ),

        },

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


        personalized = group[
            "personalized_prediction"
        ].to_numpy()


        timesfm_mae = mean_absolute_error(

            actual,

            timesfm

        )


        personalized_mae = mean_absolute_error(

            actual,

            personalized

        )


        timesfm_rmse = np.sqrt(

            mean_squared_error(

                actual,

                timesfm

            )

        )


        personalized_rmse = np.sqrt(

            mean_squared_error(

                actual,

                personalized

            )

        )


        rows.append({

            "horizon_minutes":
                int(horizon),

            "samples":
                len(group),

            "timesfm_mae":
                timesfm_mae,

            "personalized_mae":
                personalized_mae,

            "timesfm_rmse":
                timesfm_rmse,

            "personalized_rmse":
                personalized_rmse,

        })


    return pd.DataFrame(
        rows
    ).sort_values(
        "horizon_minutes"
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


    print()

    print(
        "TIMESFM + PERSONALIZATION"
    )

    print(
        "-------------------------"
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


    print()

    print(
        "IMPROVEMENT"
    )

    print(
        "-----------"
    )


    print(

        f"MAE : "
        f"{metrics['improvement']['mae_percent']:+.1f}%"

    )


    print(

        f"RMSE: "
        f"{metrics['improvement']['rmse_percent']:+.1f}%"

    )


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
        "TimesFM MAE | "
        "Personalized MAE"

    )


    print(

        "--------+---------+-------------+-----------------"

    )


    for _, row in horizon_results.iterrows():

        print(

            f"{int(row['horizon_minutes']):7d} | "

            f"{int(row['samples']):7d} | "

            f"{row['timesfm_mae']:11.3f} | "

            f"{row['personalized_mae']:15.3f}"

        )


# =========================================================
# SAVE RESULTS
# =========================================================

def save_results(
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
# TRAIN FINAL MODEL
# =========================================================

def train_final_model(
    df
):

    print()

    print(
        "Training final model on all available data..."
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
        f"Final model saved to {MODEL_FILE}"
    )


    return model


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
    # SAVE METRICS
    # -----------------------------------------------------

    save_results(

        metrics,

        horizon_results

    )


    # -----------------------------------------------------
    # FINAL MODEL
    # -----------------------------------------------------

    train_final_model(
        df
    )


    print()

    print(
        "=========================================================="
    )

    print(
        "Walk-forward training completed successfully."
    )

    print(
        "=========================================================="
    )


# =========================================================
# START
# =========================================================

if __name__ == "__main__":

    main()
