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


# Fraction of the newest data reserved for testing.
# IMPORTANT:
# We split chronologically, never randomly.

TEST_FRACTION = 0.20


# Ridge regularization.
#
# We will later tune this parameter when we have
# significantly more data.

RIDGE_ALPHA = 1.0


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


    if after < 10:

        raise RuntimeError(

            "Not enough data to train "
            "the personalization model."

        )


    return df


# =========================================================
# CHRONOLOGICAL TRAIN / TEST SPLIT
# =========================================================

def split_data(df):

    # =====================================================
    # SPLIT BY FORECAST RUN
    # =====================================================
    #
    # One TimesFM run creates many horizons:
    #
    # +5, +10, +15, ... +120 min
    #
    # All rows belonging to the same forecast run must
    # stay together. Otherwise information from one
    # forecast run could appear in both TRAIN and TEST.

    forecast_runs = (

        df[
            "forecast_created_at"
        ]

        .drop_duplicates()

        .sort_values()

        .reset_index(drop=True)

    )


    if len(forecast_runs) < 5:

        raise RuntimeError(

            "Not enough independent forecast runs "
            "for a reliable train/test split."

        )


    split_index = int(

        len(forecast_runs) *
        (1.0 - TEST_FRACTION)

    )


    split_index = max(

        1,

        min(

            split_index,

            len(forecast_runs) - 1

        )

    )


    train_runs = set(

        forecast_runs.iloc[
            :split_index
        ]

    )


    test_runs = set(

        forecast_runs.iloc[
            split_index:
        ]

    )


    train_df = (

        df[
            df["forecast_created_at"]
            .isin(train_runs)
        ]

        .copy()

        .reset_index(drop=True)

    )


    test_df = (

        df[
            df["forecast_created_at"]
            .isin(test_runs)
        ]

        .copy()

        .reset_index(drop=True)

    )


    print()

    print(
        "=========================================================="
    )

    print(
        "CHRONOLOGICAL FORECAST-RUN SPLIT"
    )

    print(
        "=========================================================="
    )


    print(
        f"Forecast runs total: "
        f"{len(forecast_runs)}"
    )


    print(
        f"Training runs:       "
        f"{len(train_runs)}"
    )


    print(
        f"Testing runs:        "
        f"{len(test_runs)}"
    )


    print()

    print(
        f"Training rows: "
        f"{len(train_df)}"
    )


    print(
        f"Testing rows:  "
        f"{len(test_df)}"
    )


    print()

    print(
        "Training period:"
    )


    print(

        f"{train_df['forecast_created_at'].min()}"

        " → "

        f"{train_df['forecast_created_at'].max()}"

    )


    print()

    print(
        "Testing period:"
    )


    print(

        f"{test_df['forecast_created_at'].min()}"

        " → "

        f"{test_df['forecast_created_at'].max()}"

    )


    return (

        train_df,

        test_df

    )


# =========================================================
# TRAIN MODEL
# =========================================================

def train_model(
    train_df
):

    print()

    print(
        "Training Ridge personalization model..."
    )


    X_train = train_df[
        FEATURE_COLUMNS
    ]


    y_train = train_df[
        TARGET_COLUMN
    ]


    # -----------------------------------------------------
    # StandardScaler + Ridge
    # -----------------------------------------------------

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


    model.fit(

        X_train,

        y_train

    )


    print(
        "Model trained successfully."
    )


    return model


# =========================================================
# EVALUATE
# =========================================================

def evaluate_model(
    model,
    df
):

    X = df[
        FEATURE_COLUMNS
    ]


    actual = df[
        "actual_glucose"
    ].to_numpy()


    timesfm = df[
        "timesfm_prediction"
    ].to_numpy()


    # -----------------------------------------------------
    # Personalization correction
    # -----------------------------------------------------

    predicted_correction = (

        model.predict(X)

    )


    personalized_prediction = (

        timesfm +
        predicted_correction

    )


    # -----------------------------------------------------
    # TimesFM baseline
    # -----------------------------------------------------

    timesfm_mae = (

        mean_absolute_error(

            actual,

            timesfm

        )

    )


    timesfm_rmse = np.sqrt(

        mean_squared_error(

            actual,

            timesfm

        )

    )


    timesfm_bias = np.mean(

        actual -
        timesfm

    )


    # -----------------------------------------------------
    # Personalized model
    # -----------------------------------------------------

    personalized_mae = (

        mean_absolute_error(

            actual,

            personalized_prediction

        )

    )


    personalized_rmse = np.sqrt(

        mean_squared_error(

            actual,

            personalized_prediction

        )

    )


    personalized_bias = np.mean(

        actual -
        personalized_prediction

    )


    metrics = {

        "samples":
            int(len(df)),

        "timesfm": {

            "mae":
                float(timesfm_mae),

            "rmse":
                float(timesfm_rmse),

            "bias":
                float(timesfm_bias),

        },

        "personalized": {

            "mae":
                float(personalized_mae),

            "rmse":
                float(personalized_rmse),

            "bias":
                float(personalized_bias),

        },

    }


    return (
        metrics,
        predicted_correction,
        personalized_prediction
    )


# =========================================================
# PRINT RESULTS
# =========================================================

def print_results(
    metrics
):

    timesfm = metrics[
        "timesfm"
    ]


    personalized = metrics[
        "personalized"
    ]


    print()

    print(
        "=========================================================="
    )

    print(
        "TEST RESULTS"
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
        f"{timesfm['mae']:.3f} mmol/l"
    )

    print(
        f"RMSE : "
        f"{timesfm['rmse']:.3f} mmol/l"
    )

    print(
        f"Bias : "
        f"{timesfm['bias']:+.3f} mmol/l"
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
        f"{personalized['mae']:.3f} mmol/l"
    )

    print(
        f"RMSE : "
        f"{personalized['rmse']:.3f} mmol/l"
    )

    print(
        f"Bias : "
        f"{personalized['bias']:+.3f} mmol/l"
    )


    print()

    mae_difference = (

        personalized["mae"]
        -
        timesfm["mae"]

    )


    rmse_difference = (

        personalized["rmse"]
        -
        timesfm["rmse"]

    )


    print(
        "DIFFERENCE"
    )

    print(
        "----------"
    )


    print(

        f"MAE difference: "
        f"{mae_difference:+.3f} mmol/l"

    )


    print(

        f"RMSE difference: "
        f"{rmse_difference:+.3f} mmol/l"

    )


    print()


# =========================================================
# SAVE MODEL
# =========================================================

def save_model(
    model
):

    joblib.dump(

        model,

        MODEL_FILE

    )


    print(
        f"Model saved to {MODEL_FILE}"
    )


# =========================================================
# SAVE METRICS
# =========================================================

def save_metrics(
    metrics
):

    with open(

        METRICS_FILE,

        "w",

        encoding="utf-8"

    ) as file:

        json.dump(

            metrics,

            file,

            indent=2

        )


    print(
        f"Metrics saved to {METRICS_FILE}"
    )


# =========================================================
# MAIN
# =========================================================

def main():

    # -----------------------------------------------------
    # Load
    # -----------------------------------------------------

    df = load_dataset()


    # -----------------------------------------------------
    # Prepare
    # -----------------------------------------------------

    df = prepare_data(
        df
    )


    # -----------------------------------------------------
    # Split
    # -----------------------------------------------------

    (
        train_df,
        test_df
    ) = split_data(
        df
    )


    # -----------------------------------------------------
    # Train
    # -----------------------------------------------------

    model = train_model(
        train_df
    )


    # -----------------------------------------------------
    # Test
    # -----------------------------------------------------

    (
        metrics,
        corrections,
        personalized
    ) = evaluate_model(

        model,

        test_df

    )


    # -----------------------------------------------------
    # Results
    # -----------------------------------------------------

    print_results(
        metrics
    )


    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    save_model(
        model
    )


    save_metrics(
        metrics
    )


    print()

    print(
        "Training completed successfully."
    )


# =========================================================
# START
# =========================================================

if __name__ == "__main__":

    main()
