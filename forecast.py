import os
import requests
import numpy as np
import pandas as pd

from timesfm3 import TimesFM3Evaluator, ModelConfig


# ============================================================
# KONFIGURACE
# ============================================================

SUPABASE_URL = "https://tpzwxutiveixprniptyh.supabase.co"
TABLE_NAME = "G_Entries"

# Kolik aktuální historie maximálně načteme
LOOKBACK_HOURS = 12

# Požadovaný interval
INTERVAL = "5min"

# Kolik bodů chceme použít pro predikci
MAX_INPUT_POINTS = 72

# Kolik bodů chceme předpovědět
FORECAST_POINTS = 12


# ============================================================
# SUPABASE
# ============================================================

API_KEY = os.getenv("SUPABASE_API_KEY")

if not API_KEY:
    raise RuntimeError("SUPABASE_API_KEY není nastavena.")


headers = {
    "apikey": API_KEY,
    "Authorization": f"Bearer {API_KEY}",
}


# ============================================================
# NAČTENÍ DAT
# ============================================================

print("Načítám aktuální data ze Supabase...")

url = (
    f"{SUPABASE_URL}/rest/v1/{TABLE_NAME}"
    "?select=dateString,sgv_mmol"
    "&order=dateString.desc"
    "&limit=5000"
)

response = requests.get(url, headers=headers)

if response.status_code != 200:
    raise RuntimeError(
        f"Supabase chyba {response.status_code}: {response.text}"
    )

data = response.json()

if not data:
    raise RuntimeError("Supabase nevrátil žádná data.")


df = pd.DataFrame(data)

df["dateString"] = pd.to_datetime(df["dateString"], utc=True)
df["sgv_mmol"] = pd.to_numeric(df["sgv_mmol"], errors="coerce")

df = df.dropna(subset=["dateString", "sgv_mmol"])
df = df.sort_values("dateString")


# ============================================================
# POUZE AKTUÁLNÍ DATA
# ============================================================

latest_time = df["dateString"].max()
cutoff_time = latest_time - pd.Timedelta(hours=LOOKBACK_HOURS)

df = df[df["dateString"] >= cutoff_time].copy()

print()
print(f"Nejnovější měření: {latest_time}")
print(f"Počet aktuálních měření: {len(df)}")


# ============================================================
# PŘEVOD NA 5MINUTOVÝ INTERVAL
# ============================================================

df = df.set_index("dateString")

series = df["sgv_mmol"].resample(INTERVAL).mean()

# Doplnění případných mezer interpolací
series = series.interpolate(method="time")

series = series.dropna()


print(f"Počet 5minutových bodů: {len(series)}")


if len(series) < 2:
    raise RuntimeError(
        "Máme příliš málo aktuálních dat pro test TimesFM."
    )


# Použijeme maximálně posledních 72 bodů
input_series = series.tail(MAX_INPUT_POINTS)

values = input_series.to_numpy(dtype=np.float32)


print()
print("========================================")
print("VSTUP PRO TIMESFM")
print("========================================")
print(f"Počet hodnot: {len(values)}")
print(f"Od: {input_series.index[0]}")
print(f"Do: {input_series.index[-1]}")

print()
print("Poslední hodnoty:")
for timestamp, value in input_series.tail(10).items():
    print(f"{timestamp}  {value:.2f} mmol/l")


# ============================================================
# TIMESFM 3.0
# ============================================================

print()
print("Načítám TimesFM 3.0...")

config = ModelConfig(
    checkpoint_path="google/timesfm-3.0-pytorch",
    per_core_batch_size=1,
    device="cpu",
)

forecaster = TimesFM3Evaluator(config)


print("Provádím predikci...")

outputs = list(
    forecaster.predict_batch(
        [values],
        horizon=FORECAST_POINTS,
        return_quantiles=True,
        use_symmetric_averaging=False,
    )
)


forecast = np.asarray(outputs[0].forecast).reshape(-1)


# ============================================================
# VÝSLEDEK
# ============================================================

last_timestamp = input_series.index[-1]

forecast_times = pd.date_range(
    start=last_timestamp + pd.Timedelta(minutes=5),
    periods=FORECAST_POINTS,
    freq=INTERVAL,
)


print()
print("========================================")
print("TIMESFM 3.0 – PREDIKCE")
print("========================================")

for timestamp, value in zip(forecast_times, forecast):
    print(f"{timestamp}  ->  {value:.2f} mmol/l")


print()
print("========================================")
print("HOTOVO")
print("========================================")
print(f"Vstup:     {len(values)} hodnot")
print(f"Predikce:  {len(forecast)} hodnot")
print(f"Horizont:  {len(forecast) * 5} minut")