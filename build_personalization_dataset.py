import os
import requests
import pandas as pd
import numpy as np


SUPABASE_URL = "https://tpzwxutiveixprniptyh.supabase.co"

GLUCOSE_TABLE = "G_Entries"

EVALUATION_TABLE = "glucose_forecast_evaluations"


# =========================================================
# SETTINGS
# =========================================================

# How close an actual CGM measurement must be to the
# forecast creation time when reconstructing the state
# that was known at prediction time.
MATCH_TOLERANCE_MINUTES = 3


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
# LOAD FORECAST EVALUATIONS
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

        # We only want forecasts for which we already
        # know the actual glucose value.

        "actual_mmol":
            "not.is.null",

        "order":
            "forecast_created_at.asc",

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

        return pd.DataFrame()


    df = pd.DataFrame(rows)


    # -----------------------------------------------------
    # Datetime conversion
    # -----------------------------------------------------

    df["forecast_created_at"] = pd.to_datetime(

        df["forecast_created_at"],

        utc=True

    )


    df["forecast_time"] = pd.to_datetime(

        df["forecast_time"],

        utc=True

    )


    # -----------------------------------------------------
    # Numeric conversion
    # -----------------------------------------------------

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

            "forecast_created_at",

            "forecast_time",

            "horizon_minutes",

            "predicted_mmol",

            "actual_mmol",

        ]

    )


    return df


# =========================================================
# LOAD GLUCOSE HISTORY
# =========================================================

def load_glucose():

    url = (
        f"{SUPABASE_URL}/rest/v1/"
        f"{GLUCOSE_TABLE}"
    )


    params = {

        "select":
            "dateString,sgv_mmol",

        # IMPORTANT:
        # newest data first

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

        return pd.DataFrame()


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

            "sgv_mmol",

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
# GET GLUCOSE AT A SPECIFIC TIME
# =========================================================

def glucose_at_time(

    glucose_df,

    timestamp,

    tolerance_minutes=MATCH_TOLERANCE_MINUTES

):

    timestamp = pd.Timestamp(
        timestamp
    )


    if timestamp.tzinfo is None:

        timestamp = (
            timestamp
            .tz_localize("UTC")
        )

    else:

        timestamp = (
            timestamp
            .tz_convert("UTC")
        )


    differences = (

        glucose_df["dateString"] -
        timestamp

    ).abs()


    if differences.empty:

        return np.nan


    idx = differences.idxmin()


    difference = differences.loc[idx]


    if difference <= pd.Timedelta(

        minutes=tolerance_minutes

    ):

        return float(

            glucose_df.loc[
                idx,
                "sgv_mmol"
            ]

        )


    return np.nan


# =========================================================
# BUILD ONE FEATURE SET
# =========================================================

def build_features_for_forecast(

    forecast_row,

    glucose_df

):

    created_at = (
        forecast_row[
            "forecast_created_at"
        ]
    )


    # -----------------------------------------------------
    # CURRENT GLUCOSE
    # -----------------------------------------------------

    current = glucose_at_time(

        glucose_df,

        created_at

    )


    if pd.isna(current):

        return None


    # -----------------------------------------------------
    # PAST GLUCOSE
    # -----------------------------------------------------

    glucose_5m = glucose_at_time(

        glucose_df,

        created_at -
        pd.Timedelta(minutes=5)

    )


    glucose_15m = glucose_at_time(

        glucose_df,

        created_at -
        pd.Timedelta(minutes=15)

    )


    glucose_30m = glucose_at_time(

        glucose_df,

        created_at -
        pd.Timedelta(minutes=30)

    )


    glucose_60m = glucose_at_time(

        glucose_df,

        created_at -
        pd.Timedelta(minutes=60)

    )


    # -----------------------------------------------------
    # CHANGES
    # -----------------------------------------------------

    if pd.notna(glucose_5m):

        delta_5m = (
            current -
            glucose_5m
        )

    else:

        delta_5m = np.nan


    if pd.notna(glucose_15m):

        delta_15m = (
            current -
            glucose_15m
        )

    else:

        delta_15m = np.nan


    if pd.notna(glucose_30m):

        delta_30m = (
            current -
            glucose_30m
        )

    else:

        delta_30m = np.nan


    if pd.notna(glucose_60m):

        delta_60m = (
            current -
            glucose_60m
        )

    else:

        delta_60m = np.nan


    # -----------------------------------------------------
    # TIME
    # -----------------------------------------------------

    hour = created_at.hour

    minute = created_at.minute


    hour_sin = np.sin(

        2 *
        np.pi *
        hour /
        24

    )


    hour_cos = np.cos(

        2 *
        np.pi *
        hour /
        24

    )


    # -----------------------------------------------------
    # TARGET
    # -----------------------------------------------------

    actual = float(

        forecast_row[
            "actual_mmol"
        ]

    )


    prediction = float(

        forecast_row[
            "predicted_mmol"
        ]

    )


    correction_target = (

        actual -
        prediction

    )


    # -----------------------------------------------------
    # RESULT
    # -----------------------------------------------------

    return {

        "forecast_created_at":
            created_at.isoformat(),

        "forecast_time":
            forecast_row[
                "forecast_time"
            ].isoformat(),

        "horizon_minutes":
            int(
                forecast_row[
                    "horizon_minutes"
                ]
            ),

        # -------------------------------------------------
        # FEATURES
        # -------------------------------------------------

        "current_glucose":
            current,

        "glucose_5m":
            glucose_5m,

        "glucose_15m":
            glucose_15m,

        "glucose_30m":
            glucose_30m,

        "glucose_60m":
            glucose_60m,

        "delta_5m":
            delta_5m,

        "delta_15m":
            delta_15m,

        "delta_30m":
            delta_30m,

        "delta_60m":
            delta_60m,

        "timesfm_prediction":
            prediction,

        "hour":
            hour,

        "minute":
            minute,

        "hour_sin":
            hour_sin,

        "hour_cos":
            hour_cos,

        # -------------------------------------------------
        # TARGET
        # -------------------------------------------------

        "actual_glucose":
            actual,

        "correction_target":
            correction_target,

    }


# =========================================================
# BUILD DATASET
# =========================================================

def build_dataset(

    evaluations,

    glucose_df

):

    rows = []


    for index, forecast in (
        evaluations.iterrows()
    ):

        features = (
            build_features_for_forecast(
                forecast,
                glucose_df
            )
        )


        if features is not None:

            rows.append(
                features
            )


        if (
            (index + 1) % 100 == 0
        ):

            print(

                f"Processed "
                f"{index + 1} / "
                f"{len(evaluations)} "
                f"evaluations"

            )


    if not rows:

        return pd.DataFrame()


    dataset = pd.DataFrame(
        rows
    )


    return dataset


# =========================================================
# REMOVE INCOMPLETE ROWS
# =========================================================

def clean_dataset(dataset):

    required_columns = [

        "current_glucose",

        "delta_5m",

        "delta_15m",

        "delta_30m",

        "timesfm_prediction",

        "horizon_minutes",

        "actual_glucose",

        "correction_target",

    ]


    before = len(dataset)


    dataset = dataset.dropna(

        subset=required_columns

    )


    after = len(dataset)


    print()

    print(
        f"Rows before cleaning: {before}"
    )

    print(
        f"Rows after cleaning:  {after}"
    )

    print(
        f"Rows removed:         "
        f"{before - after}"
    )


    return dataset


# =========================================================
# SAVE DATASET
# =========================================================

def save_dataset(dataset):

    output_file = "personalization_dataset.csv"

    dataset.to_csv(

        output_file,

        index=False

    )


    print()

    print(
        "Dataset saved to:"
    )

    print(
        output_file
    )


# =========================================================
# PRINT PREVIEW
# =========================================================

def print_preview(dataset):

    if dataset.empty:

        print(
            "Dataset is empty."
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

    columns = [

        "horizon_minutes",

        "current_glucose",

        "delta_5m",

        "delta_15m",

        "delta_30m",

        "timesfm_prediction",

        "actual_glucose",

        "correction_target",

    ]


    print(

        dataset[columns]
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


    evaluations = load_evaluations()


    if evaluations.empty:

        print(
            "No evaluated forecasts available."
        )

        return


    print(

        f"Loaded "
        f"{len(evaluations)} "
        "evaluated forecasts."

    )


    print(
        "Loading glucose history..."
    )


    glucose_df = load_glucose()


    if glucose_df.empty:

        print(
            "No glucose data available."
        )

        return


    print(

        f"Loaded "
        f"{len(glucose_df)} "
        "glucose measurements."

    )


    print(

        "Glucose range: "

        f"{glucose_df['dateString'].min().isoformat()}"

        " → "

        f"{glucose_df['dateString'].max().isoformat()}"

    )


    # -----------------------------------------------------
    # BUILD
    # -----------------------------------------------------

    dataset = build_dataset(

        evaluations,

        glucose_df

    )


    if dataset.empty:

        print(
            "Could not build dataset."
        )

        return


    # -----------------------------------------------------
    # CLEAN
    # -----------------------------------------------------

    dataset = clean_dataset(
        dataset
    )


    if dataset.empty:

        print(
            "No complete rows remain."
        )

        return


    # -----------------------------------------------------
    # SAVE
    # -----------------------------------------------------

    save_dataset(
        dataset
    )


    # -----------------------------------------------------
    # PREVIEW
    # -----------------------------------------------------

    print_preview(
        dataset
    )


    print()

    print(
        "Dataset creation completed successfully."
    )


# =========================================================
# START
# =========================================================

if __name__ == "__main__":

    main()
