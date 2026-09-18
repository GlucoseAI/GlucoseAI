import os

import pandas as pd
import requests


SUPABASE_URL = "https://tpzwxutiveixprniptyh.supabase.co"

GLUCOSE_TABLE = "G_Entries"

EVALUATION_TABLE = "glucose_forecast_evaluations"

# Jak daleko od forecast_time smíme hledat
# skutečné měření.
MATCH_TOLERANCE_MINUTES = 3


# =========================================================
# SUPABASE HEADERS
# =========================================================

def get_headers():

    key = os.environ.get("SUPABASE_API_KEY")

    if not key:

        raise RuntimeError(
            "SUPABASE_API_KEY is not set"
        )

    return {

        "apikey": key,

        "Authorization":
            f"Bearer {key}",

        "Content-Type":
            "application/json",

    }


# =========================================================
# PENDING FORECASTS
# =========================================================

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

        "actual_mmol":
            "is.null",

        "order":
            "forecast_time.asc",

        "limit":
            "5000",

    }


    response = requests.get(

        url,

        headers=get_headers(),

        params=params,

        timeout=30,

    )


    response.raise_for_status()


    return response.json()


# =========================================================
# GLUCOSE DATA
# =========================================================

def fetch_glucose_data():

    url = (

        f"{SUPABASE_URL}/rest/v1/"
        f"{GLUCOSE_TABLE}"

    )


    params = {

        "select":
            "dateString,sgv_mmol",

        # DŮLEŽITÉ:
        # vezmeme nejnovější data,
        # ne nejstarší data.

        "order":
            "dateString.desc",

        "limit":
            "5000",

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


    df = (

        df

        .sort_values(
            "dateString"
        )

        .drop_duplicates(
            "dateString"
        )

    )


    return df


# =========================================================
# FIND ACTUAL GLUCOSE
# =========================================================

def find_actual_value(

    forecast_time,

    glucose_df

):

    forecast_time = pd.Timestamp(

        forecast_time

    )


    if forecast_time.tzinfo is None:

        forecast_time = (

            forecast_time
            .tz_localize("UTC")

        )

    else:

        forecast_time = (

            forecast_time
            .tz_convert("UTC")

        )


    differences = (

        glucose_df["dateString"] -
        forecast_time

    ).abs()


    if differences.empty:

        return None


    closest_index = (

        differences.idxmin()

    )


    closest_difference = (

        differences.loc[
            closest_index
        ]

    )


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


# =========================================================
# UPDATE EVALUATION
# =========================================================

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

        "id":
            f"eq.{evaluation_id}"

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

    headers["Prefer"] = (
        "return=minimal"
    )


    response = requests.patch(

        url,

        headers=headers,

        params=params,

        json=data,

        timeout=30,

    )


    response.raise_for_status()


# =========================================================
# MAIN
# =========================================================

def main():

    print(
        "Loading pending forecast evaluations..."
    )


    forecasts = (
        fetch_pending_forecasts()
    )


    if not forecasts:

        print(
            "No pending forecast evaluations."
        )

        return


    print(

        f"Found {len(forecasts)} "
        "pending forecasts."

    )


    # -----------------------------------------------------
    # Load CURRENT glucose data
    # -----------------------------------------------------

    glucose_df = (
        fetch_glucose_data()
    )


    if glucose_df.empty:

        print(
            "No glucose data available."
        )

        return


    print(

        "Latest glucose measurement: "

        f"{glucose_df['dateString'].max().isoformat()}"

    )


    print(

        "Oldest loaded glucose measurement: "

        f"{glucose_df['dateString'].min().isoformat()}"

    )


    updated = 0

    waiting = 0

    no_match = 0


    now = pd.Timestamp.now(
        tz="UTC"
    )


    # =====================================================
    # EVALUATE FORECASTS
    # =====================================================

    for forecast in forecasts:

        forecast_id = (
            forecast["id"]
        )


        forecast_time = pd.Timestamp(

            forecast["forecast_time"]

        )


        predicted = float(

            forecast["predicted_mmol"]

        )


        # -------------------------------------------------
        # Forecast is still in the future
        # -------------------------------------------------

        if forecast_time > now:

            waiting += 1

            continue


        # -------------------------------------------------
        # Find actual glucose
        # -------------------------------------------------

        actual = find_actual_value(

            forecast_time,

            glucose_df

        )


        if actual is None:

            no_match += 1

            continue


        # -------------------------------------------------
        # Save result
        # -------------------------------------------------

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

            f"horizon forecast={predicted:.3f}, "

            f"actual={actual:.3f}, "

            f"error={error:+.3f}"

        )


        updated += 1


    # =====================================================
    # SUMMARY
    # =====================================================

    print()

    print(
        "======================================"
    )

    print(
        "EVALUATION SUMMARY"
    )

    print(
        "======================================"
    )

    print(
        f"Updated:    {updated}"
    )

    print(
        f"Future:     {waiting}"
    )

    print(
        f"No match:   {no_match}"
    )

    print(
        f"Pending:    {len(forecasts)}"
    )

    print(
        "======================================"
    )


# =========================================================
# START
# =========================================================

if __name__ == "__main__":

    main()
