import os
from datetime import timedelta

import numpy as np
import pandas as pd
import requests
from timesfm3 import TimesFM3Evaluator, ModelConfig

SUPABASE_URL = "https://tpzwxutiveixprniptyh.supabase.co"
TABLE_NAME = "G_Entries"
FORECAST_TABLE = "glucose_forecasts"

# Use a much longer context: 24 hours at 5-minute resolution.
LOOKBACK_HOURS = 24
INTERVAL = "5min"
INPUT_POINTS = 288          # 24 h / 5 min
FORECAST_POINTS = 24        # 2 h / 5 min


def get_headers():
    key = os.environ.get("SUPABASE_API_KEY")
    if not key:
        raise RuntimeError("SUPABASE_API_KEY is not set")
    return {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }


def fetch_glucose():
    url = f"{SUPABASE_URL}/rest/v1/{TABLE_NAME}"
    params = {
        "select": "dateString,sgv_mmol",
        "order": "dateString.desc",
        "limit": "10000",
    }
    r = requests.get(url, headers=get_headers(), params=params, timeout=30)
    r.raise_for_status()

    rows = r.json()
    if not rows:
        raise RuntimeError("No glucose data returned from Supabase")

    df = pd.DataFrame(rows)
    df["dateString"] = pd.to_datetime(df["dateString"], utc=True)
    df["sgv_mmol"] = pd.to_numeric(df["sgv_mmol"], errors="coerce")
    df = df.dropna(subset=["dateString", "sgv_mmol"])
    df = df.sort_values("dateString").drop_duplicates("dateString")
    return df


def prepare_input(df):
    latest = df["dateString"].max()
    start = latest - pd.Timedelta(hours=LOOKBACK_HOURS)

    df = df[df["dateString"] >= start].copy()
    if df.empty:
        raise RuntimeError("No glucose data in the requested lookback window")

    # Put irregular CGM measurements onto a regular 5-minute grid.
    series = (
        df.set_index("dateString")["sgv_mmol"]
        .resample(INTERVAL)
        .mean()
        .interpolate(method="time")
        .ffill()
        .bfill()
    )

    if len(series) < INPUT_POINTS:
        print(
            f"Only {len(series)} regular 5-minute points are available. "
            f"TimesFM will use all available points until 24 h is collected."
        )
        values = series.to_numpy(dtype=np.float32)
    else:
        values = series.iloc[-INPUT_POINTS:].to_numpy(dtype=np.float32)

    if len(values) < 8:
        raise RuntimeError(
            f"Not enough current contiguous data for forecasting: {len(values)} points"
        )

    last_grid_time = series.index[-1]
    return values, last_grid_time, len(series)


def run_timesfm(values):
    config = ModelConfig(
        checkpoint_path="google/timesfm-3.0-pytorch",
        per_core_batch_size=1,
        device="cpu",
    )
    forecaster = TimesFM3Evaluator(config)

    outputs = list(
        forecaster.predict_batch(
            [values],
            horizon=FORECAST_POINTS,
            return_quantiles=True,
            use_symmetric_averaging=False,
        )
    )
    out = outputs[0]

    forecast = np.asarray(out.forecast).reshape(-1)
    quantiles = np.asarray(out.quantiles)

    # TimesFM 3.0 returns 9 quantiles: 0.1 ... 0.9.
    # For a univariate series, shape is (horizon, 9).
    if quantiles.ndim == 3:
        quantiles = quantiles[0]

    p10 = quantiles[:, 0]
    p90 = quantiles[:, 8]

    return forecast, p10, p90


def save_forecast(last_grid_time, forecast, p10, p90):
    headers = get_headers()

    # Keep only the newest forecast run in the public table.
    delete_url = f"{SUPABASE_URL}/rest/v1/{FORECAST_TABLE}"
    delete_headers = dict(headers)
    delete_headers["Prefer"] = "return=minimal"

    # id=gt.0 matches every normal identity row and avoids depending on a
    # particular primary-key value.
    r = requests.delete(
        delete_url,
        headers=delete_headers,
        params={"id": "gt.0"},
        timeout=30,
    )
    r.raise_for_status()

    rows = []
    for i, (pred, lo, hi) in enumerate(zip(forecast, p10, p90), start=1):
        ts = last_grid_time + timedelta(minutes=5 * i)
        rows.append(
            {
                "forecast_time": ts.isoformat(),
                "predicted_mmol": float(pred),
                "p10_mmol": float(lo),
                "p90_mmol": float(hi),
            }
        )

    post_url = f"{SUPABASE_URL}/rest/v1/{FORECAST_TABLE}"
    post_headers = dict(headers)
    post_headers["Prefer"] = "return=minimal"

    r = requests.post(
        post_url,
        headers=post_headers,
        json=rows,
        timeout=30,
    )
    if r.status_code not in (200, 201):
        raise RuntimeError(
            f"Forecast insert failed: HTTP {r.status_code} - {r.text}"
        )

    return len(rows)


def main():
    df = fetch_glucose()
    values, last_grid_time, regular_points = prepare_input(df)

    print(f"Latest source measurement: {df['dateString'].max().isoformat()}")
    print(f"Regular 5-minute points available: {regular_points}")
    print(f"TimesFM context points used: {len(values)}")
    print(f"Forecast horizon: {FORECAST_POINTS} points / 2 hours")

    forecast, p10, p90 = run_timesfm(values)

    saved = save_forecast(last_grid_time, forecast, p10, p90)

    print("FORECAST OK")
    for i, (pred, lo, hi) in enumerate(zip(forecast, p10, p90), start=1):
        ts = last_grid_time + timedelta(minutes=5 * i)
        print(
            f"{ts.isoformat()} | median={pred:.3f} | "
            f"P10={lo:.3f} | P90={hi:.3f}"
        )
    print(f"Saved {saved} forecast rows to Supabase.")


if __name__ == "__main__":
    main()
