import os
from datetime import timedelta

import pandas as pd
import requests


SUPABASE_URL = "https://tpzwxutiveixprniptyh.supabase.co"

GLUCOSE_TABLE = "G_Entries"

EVALUATION_TABLE = "glucose_forecast_evaluations"


# Jak daleko od forecast_time smíme hledat skutečné měření.
# CGM data jsou typicky po 5 minutách, takže 3 minuty
# poskytují malou toleranci časového posunu.
MATCH_TOLERANCE_MINUTES = 3


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


def fetch_pending_forecasts():

    url = (
        f"{SUPABASE_URL}/rest/v1/"
        f"{EVALUATION_TABLE}"
    )

    params = {
        "select": (
            "id,"
            "forecast_time,"
            "predicted_mmol"
        ),

        "actual_mmol": "is.null",

        "order": "forecast_time.asc",

        "limit": "5000",
    }

    response = requests.get(
        url,
        headers=get_headers(),
        params=params,
        timeout=30,
    )

    response.raise_for_status()

    return response.json()


def fetch_glucose_data():

    url = (
        f"{SUPABASE_URL}/rest/v1/"
        f"{GLUCOSE_TABLE}"
    )

    params = {
        "select": "dateString,sgv_mmol",

        "order": "dateString.asc",

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
        return pd.DataFrame(
            columns=[
                "dateString",
                "sgv_mmol"
            ]
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

    return df


def find_actual_value(
    forecast_time,
    glucose_df
):

    forecast_time = pd.Timestamp(
        forecast_time
    )

    if forecast_time.tzinfo is None:

        forecast_time = forecast_time.tz_localize(
            "UTC"
        )

    else:

        forecast_time = forecast_time.tz_convert(
            "UTC"
        )


    differences = (

        glucose_df["dateString"] -
        forecast_time

    ).abs()


    if differences.empty:

        return None


    closest_index = differences.idxmin()

    closest_difference = differences.loc[
        closest_index
    ]


    if closest_difference <= pd.Timedelta(
        minutes=MATCH_TOLERANCE_MINUTES
    ):

        return float(
            glucose_df.loc[
                closest_index,
                "sgv_mmol"
            ]
        )


    return None


def update_evaluation(
    evaluation_id,
    actual_mmol,
    predicted_mmol
):

    error = (
        actual_mmol -
        predicted_mmol
    )

    absolute_error = abs(error)


    url = (
        f"{SUPABASE_URL}/rest/v1/"
        f"{EVALUATION_TABLE}"
    )


    params = {
        "id": f"eq.{evaluation_id}"
    }


    data = {

        "actual_mmol":
            actual_mmol,

        "error_mmol":
            error,

        "absolute_error_mmol":
            absolute_error,

    }


    headers = get_headers()

    headers["Prefer"] = "return=minimal"


    response = requests.patch(

        url,

        headers=headers,

        params=params,

        json=data,

        timeout=30,

    )


    response.raise_for_status()


def main():

    print(
        "Loading pending forecast evaluations..."
    )


    forecasts = fetch_pending_forecasts()


    if not forecasts:

        print(
            "No pending forecast evaluations."
        )

        return


    print(
        f"Found {len(forecasts)} "
        "pending forecasts."
    )


    glucose_df = fetch_glucose_data()


    if glucose_df.empty:

        print(
            "No glucose data available."
        )

        return


    updated = 0

    waiting = 0


    for forecast in forecasts:

        forecast_id = forecast["id"]

        forecast_time = pd.Timestamp(
            forecast["forecast_time"]
        )

        predicted = float(
            forecast["predicted_mmol"]
        )


        # Pokud je forecast v budoucnosti,
        # nemáme ho ještě čím vyhodnotit.

        now = pd.Timestamp.now(
            tz="UTC"
        )


        if forecast_time > now:

            waiting += 1

            continue


        actual = find_actual_value(

            forecast_time,

            glucose_df

        )


        if actual is None:

            waiting += 1

            continue


        update_evaluation(

            forecast_id,

            actual,

            predicted

        )


        error = (
            actual -
            predicted
        )


        print(

            f"Evaluated ID {forecast_id}: "

            f"forecast={predicted:.3f}, "

            f"actual={actual:.3f}, "

            f"error={error:+.3f}"

        )


        updated += 1


    print(
        f"Updated evaluations: {updated}"
    )

    print(
        f"Still waiting: {waiting}"
    )


if __name__ == "__main__":

    main()
