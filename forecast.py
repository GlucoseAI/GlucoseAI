import os
from datetime import timedelta, datetime, timezone

import numpy as np
import pandas as pd
import requests
from timesfm3 import TimesFM3Evaluator, ModelConfig


SUPABASE_URL = "https://tpzwxutiveixprniptyh.supabase.co"

TABLE_NAME = "G_Entries"

FORECAST_TABLE = "glucose_forecasts"

EVALUATION_TABLE = "glucose_forecast_evaluations"


# =========================================================
# TIMESFM SETTINGS
# =========================================================

# 24 hours of historical glucose data
LOOKBACK_HOURS = 24

# Glucose data are processed on a 5-minute grid
INTERVAL = "5min"

# 24 h / 5 min = 288 points
INPUT_POINTS = 288

# 2 h / 5 min = 24 predictions
FORECAST_POINTS = 24


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
# LOAD GLUCOSE
# =========================================================

def fetch_glucose():

    url = (
        f"{SUPABASE_URL}/rest/v1/"
        f"{TABLE_NAME}"
    )

    params = {

        "select":
            "dateString,sgv_mmol",

        "order":
            "dateString.desc",

        "limit":
            "10000",
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

        raise RuntimeError(
            "No glucose data returned from Supabase"
        )

    df = pd.DataFrame(rows)

    df["dateString"] = pd.to_datetime(
        df["dateString"],
        utc=True
    )

    df["sgv_mmol"] = pd.to_numeric(
        df["sgv_mmol"],
        errors="coerce"
    )

    df = df.dropna(
        subset=[
            "dateString",
            "sgv_mmol"
        ]
    )

    df = (
        df
        .sort_values("dateString")
        .drop_duplicates("dateString")
    )

    return df


# =========================================================
# PREPARE TIMESFM INPUT
# =========================================================

def prepare_input(df):

    latest = df["dateString"].max()

    start = (
        latest -
        pd.Timedelta(
            hours=LOOKBACK_HOURS
        )
    )

    df = df[
        df["dateString"] >= start
    ].copy()

    if df.empty:

        raise RuntimeError(
            "No glucose data in the requested "
            "lookback window"
        )


    # Convert irregular CGM measurements
    # to a regular 5-minute grid.

    series = (

        df
        .set_index("dateString")["sgv_mmol"]
        .resample(INTERVAL)
        .mean()
        .interpolate(method="time")
        .ffill()
        .bfill()

    )


    if len(series) < INPUT_POINTS:

        print(

            f"Only {len(series)} regular "
            f"5-minute points are available. "

            f"TimesFM will use all available "
            f"points until 24 h is collected."

        )

        values = series.to_numpy(
            dtype=np.float32
        )

    else:

        values = (

            series
            .iloc[-INPUT_POINTS:]
            .to_numpy(
                dtype=np.float32
            )

        )


    if len(values) < 8:

        raise RuntimeError(

            "Not enough current contiguous "
            f"data for forecasting: {len(values)} points"

        )


    last_grid_time = series.index[-1]

    return (
        values,
        last_grid_time,
        len(series)
    )


# =========================================================
# RUN TIMESFM 3
# =========================================================

def run_timesfm(values):

    config = ModelConfig(

        checkpoint_path=
            "google/timesfm-3.0-pytorch",

        per_core_batch_size=1,

        device="cpu",

    )


    forecaster = TimesFM3Evaluator(
        config
    )


    outputs = list(

        forecaster.predict_batch(

            [values],

            horizon=FORECAST_POINTS,

            return_quantiles=True,

            use_symmetric_averaging=False,

        )

    )


    out = outputs[0]


    forecast = np.asarray(
        out.forecast
    ).reshape(-1)


    quantiles = np.asarray(
        out.quantiles
    )


    # TimesFM 3 returns 9 quantiles:
    # 0.1 ... 0.9

    if quantiles.ndim == 3:

        quantiles = quantiles[0]


    p10 = quantiles[:, 0]

    p90 = quantiles[:, 8]


    return (
        forecast,
        p10,
        p90
    )


# =========================================================
# SAVE FORECAST
# =========================================================

def save_forecast(
    last_grid_time,
    forecast,
    p10,
    p90
):

    headers = get_headers()


    # -----------------------------------------------------
    # Time when this TimesFM run was created
    # -----------------------------------------------------

    forecast_created_at = (
        datetime.now(
            timezone.utc
        ).isoformat()
    )


    # =====================================================
    # 1. UPDATE PUBLIC FORECAST TABLE
    # =====================================================

    # Keep only the newest forecast run
    # in glucose_forecasts.

    delete_url = (
        f"{SUPABASE_URL}/rest/v1/"
        f"{FORECAST_TABLE}"
    )


    delete_headers = dict(headers)

    delete_headers["Prefer"] = (
        "return=minimal"
    )


    response = requests.delete(

        delete_url,

        headers=delete_headers,

        params={
            "id": "gt.0"
        },

        timeout=30,

    )


    response.raise_for_status()


    # -----------------------------------------------------
    # Create rows for current dashboard
    # -----------------------------------------------------

    forecast_rows = []


    for i, (pred, lo, hi) in enumerate(

        zip(
            forecast,
            p10,
            p90
        ),

        start=1

    ):

        forecast_time = (

            last_grid_time +

            timedelta(
                minutes=5 * i
            )

        )


        forecast_rows.append({

            "forecast_time":
                forecast_time.isoformat(),

            "predicted_mmol":
                float(pred),

            "p10_mmol":
                float(lo),

            "p90_mmol":
                float(hi),

        })


    post_url = (
        f"{SUPABASE_URL}/rest/v1/"
        f"{FORECAST_TABLE}"
    )


    post_headers = dict(headers)

    post_headers["Prefer"] = (
        "return=minimal"
    )


    response = requests.post(

        post_url,

        headers=post_headers,

        json=forecast_rows,

        timeout=30,

    )


    if response.status_code not in (
        200,
        201
    ):

        raise RuntimeError(

            "Forecast insert failed: "
            f"HTTP {response.status_code} - "
            f"{response.text}"

        )


    # =====================================================
    # 2. SAVE HISTORICAL EVALUATION ROWS
    # =====================================================

    # These rows are NEVER deleted.
    #
    # Later evaluate_forecasts.py will fill:
    #
    # actual_mmol
    # error_mmol
    # absolute_error_mmol

    evaluation_rows = []


    for i, pred in enumerate(

        forecast,

        start=1

    ):

        forecast_time = (

            last_grid_time +

            timedelta(
                minutes=5 * i
            )

        )


        evaluation_rows.append({

            "forecast_created_at":
                forecast_created_at,

            "forecast_time":
                forecast_time.isoformat(),

            "horizon_minutes":
                i * 5,

            "predicted_mmol":
                float(pred),

        })


    evaluation_url = (

        f"{SUPABASE_URL}/rest/v1/"
        f"{EVALUATION_TABLE}"

    )


    evaluation_headers = dict(headers)

    evaluation_headers["Prefer"] = (
        "return=minimal"
    )


    response = requests.post(

        evaluation_url,

        headers=evaluation_headers,

        json=evaluation_rows,

        timeout=30,

    )


    if response.status_code not in (
        200,
        201
    ):

        raise RuntimeError(

            "Evaluation insert failed: "
            f"HTTP {response.status_code} - "
            f"{response.text}"

        )


    return len(forecast_rows)


# =========================================================
# MAIN
# =========================================================

def main():

    # -----------------------------------------------------
    # Load glucose
    # -----------------------------------------------------

    df = fetch_glucose()


    # -----------------------------------------------------
    # Prepare TimesFM input
    # -----------------------------------------------------

    (
        values,
        last_grid_time,
        regular_points
    ) = prepare_input(df)


    print(
        "Latest source measurement: "
        f"{df['dateString'].max().isoformat()}"
    )


    print(
        "Regular 5-minute points available: "
        f"{regular_points}"
    )


    print(
        "TimesFM context points used: "
        f"{len(values)}"
    )


    print(
        "Forecast horizon: "
        f"{FORECAST_POINTS} points / 2 hours"
    )


    # -----------------------------------------------------
    # TimesFM
    # -----------------------------------------------------

    (
        forecast,
        p10,
        p90
    ) = run_timesfm(values)


    # -----------------------------------------------------
    # Save forecast + evaluation dataset
    # -----------------------------------------------------

    saved = save_forecast(

        last_grid_time,

        forecast,

        p10,

        p90

    )


    print(
        "FORECAST OK"
    )


    # -----------------------------------------------------
    # Print predictions
    # -----------------------------------------------------

    for i, (pred, lo, hi) in enumerate(

        zip(
            forecast,
            p10,
            p90
        ),

        start=1

    ):

        forecast_time = (

            last_grid_time +

            timedelta(
                minutes=5 * i
            )

        )


        print(

            f"{forecast_time.isoformat()} | "

            f"median={pred:.3f} | "

            f"P10={lo:.3f} | "

            f"P90={hi:.3f}"

        )


    print(
        f"Saved {saved} forecast rows "
        "to Supabase."
    )


    print(
        f"Saved {saved} evaluation rows "
        "to Supabase."
    )


# =========================================================
# START
# =========================================================

if __name__ == "__main__":

    main()
