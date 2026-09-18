import os
import requests
import pandas as pd
import numpy as np


SUPABASE_URL = "https://tpzwxutiveixprniptyh.supabase.co"

EVALUATION_TABLE = "glucose_forecast_evaluations"


# =========================================================
# SUPABASE
# =========================================================

def get_headers():

    key = os.environ.get("SUPABASE_API_KEY")

    if not key:
        raise RuntimeError(
            "SUPABASE_API_KEY is not set"
        )

    return {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }


# =========================================================
# LOAD EVALUATIONS
# =========================================================

def load_evaluations():

    url = (
        f"{SUPABASE_URL}/rest/v1/"
        f"{EVALUATION_TABLE}"
    )

    params = {
        "select": (
            "id,"
            "forecast_created_at,"
            "forecast_time,"
            "horizon_minutes,"
            "predicted_mmol,"
            "actual_mmol,"
            "error_mmol,"
            "absolute_error_mmol"
        ),

        "actual_mmol": "not.is.null",

        "order": "forecast_time.asc",

        "limit": "10000",
    }

    response = requests.get(
        url,
        headers=get_headers(),
        params=params,
        timeout=30,
    )

    response.raise_for_status()

    rows = response.json()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    return df


# =========================================================
# PREPARE DATA
# =========================================================

def prepare_data(df):

    numeric_columns = [
        "horizon_minutes",
        "predicted_mmol",
        "actual_mmol",
        "error_mmol",
        "absolute_error_mmol",
    ]

    for column in numeric_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    df = df.dropna(
        subset=[
            "horizon_minutes",
            "predicted_mmol",
            "actual_mmol",
            "error_mmol",
        ]
    )

    return df


# =========================================================
# CALCULATE METRICS
# =========================================================

def calculate_metrics(df):

    results = []

    grouped = df.groupby(
        "horizon_minutes"
    )

    for horizon, group in grouped:

        errors = group[
            "error_mmol"
        ].to_numpy()

        absolute_errors = np.abs(
            errors
        )

        mae = np.mean(
            absolute_errors
        )

        rmse = np.sqrt(
            np.mean(
                errors ** 2
            )
        )

        bias = np.mean(
            errors
        )

        results.append({

            "horizon_minutes":
                int(horizon),

            "samples":
                len(group),

            "mae":
                mae,

            "rmse":
                rmse,

            "bias":
                bias,

        })

    result_df = pd.DataFrame(
        results
    )

    if not result_df.empty:

        result_df = result_df.sort_values(
            "horizon_minutes"
        )

    return result_df


# =========================================================
# PRINT METRICS
# =========================================================

def print_metrics(metrics):

    print()

    print(
        "=========================================================="
    )

    print(
        "TIMESFM FORECAST ACCURACY"
    )

    print(
        "=========================================================="
    )

    print(
        " Horizon | Samples |   MAE |  RMSE |   Bias"
    )

    print(
        "---------+---------+-------+-------+-------"
    )


    for _, row in metrics.iterrows():

        print(

            f" {int(row['horizon_minutes']):7d} | "

            f"{int(row['samples']):7d} | "

            f"{row['mae']:5.3f} | "

            f"{row['rmse']:5.3f} | "

            f"{row['bias']:+5.3f}"

        )


    print(
        "=========================================================="
    )


# =========================================================
# OVERALL METRICS
# =========================================================

def print_overall_metrics(df):

    if df.empty:
        return

    errors = (
        df["error_mmol"]
        .to_numpy()
    )

    mae = np.mean(
        np.abs(errors)
    )

    rmse = np.sqrt(
        np.mean(
            errors ** 2
        )
    )

    bias = np.mean(
        errors
    )


    print()

    print(
        "OVERALL"
    )

    print(
        "-------"
    )

    print(
        f"Samples : {len(df)}"
    )

    print(
        f"MAE     : {mae:.3f} mmol/l"
    )

    print(
        f"RMSE    : {rmse:.3f} mmol/l"
    )

    print(
        f"Bias    : {bias:+.3f} mmol/l"
    )


# =========================================================
# DATASET FOR PERSONALIZATION
# =========================================================

def create_personalization_dataset(df):

    if df.empty:
        return pd.DataFrame()


    dataset = pd.DataFrame({

        "forecast_time":
            df["forecast_time"],

        "horizon_minutes":
            df["horizon_minutes"],

        "timesfm_prediction":
            df["predicted_mmol"],

        "actual_glucose":
            df["actual_mmol"],

        "correction_target":
            df["actual_mmol"] -
            df["predicted_mmol"],

    })


    dataset["forecast_time"] = pd.to_datetime(
        dataset["forecast_time"],
        utc=True
    )


    # -----------------------------------------------------
    # TIME FEATURES
    # -----------------------------------------------------

    dataset["hour"] = (
        dataset["forecast_time"]
        .dt.hour
    )


    dataset["minute"] = (
        dataset["forecast_time"]
        .dt.minute
    )


    # -----------------------------------------------------
    # CYCLIC TIME FEATURES
    # -----------------------------------------------------

    dataset["hour_sin"] = np.sin(

        2 *
        np.pi *
        dataset["hour"] /
        24

    )


    dataset["hour_cos"] = np.cos(

        2 *
        np.pi *
        dataset["hour"] /
        24

    )


    return dataset


# =========================================================
# DATASET PREVIEW
# =========================================================

def print_dataset_preview(dataset):

    if dataset.empty:

        print(
            "No personalization data available."
        )

        return


    print()

    print(
        "=========================================================="
    )

    print(
        "PERSONALIZATION DATASET"
    )

    print(
        "=========================================================="
    )

    print(
        f"Rows: {len(dataset)}"
    )

    print()


    print(
        dataset[
            [
                "horizon_minutes",
                "timesfm_prediction",
                "actual_glucose",
                "correction_target",
                "hour",
            ]
        ]
        .tail(10)
        .to_string(
            index=False
        )
    )


# =========================================================
# MAIN
# =========================================================

def main():

    print(
        "Loading forecast evaluations..."
    )


    df = load_evaluations()


    if df.empty:

        print(
            "No evaluated forecasts available yet."
        )

        return


    print(
        f"Loaded {len(df)} evaluated forecasts."
    )


    df = prepare_data(df)


    if df.empty:

        print(
            "No valid evaluation rows."
        )

        return


    # -----------------------------------------------------
    # METRICS
    # -----------------------------------------------------

    metrics = calculate_metrics(
        df
    )


    print_metrics(
        metrics
    )


    print_overall_metrics(
        df
    )


    # -----------------------------------------------------
    # PERSONALIZATION DATASET
    # -----------------------------------------------------

    dataset = create_personalization_dataset(
        df
    )


    print_dataset_preview(
        dataset
    )


    print()

    print(
        "Analysis completed successfully."
    )


# =========================================================
# START
# =========================================================

if __name__ == "__main__":

    main()
