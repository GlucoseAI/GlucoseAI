import json
import requests
from datetime import datetime
import os

# Supabase konfigurace
SUPABASE_URL = "https://tpzwxutiveixprniptyh.supabase.co"  # Nová URL Supabase
SUPABASE_API_KEY = os.getenv("SUPABASE_API_KEY")
print("SUPABASE_API_KEY:", SUPABASE_API_KEY)  # Debug: Zkontroluje, zda je klíč načten
TABLE_NAME = "G_Entries"  # Název tabulky

# API URL pro získání dat
#API_URL = "https://2098.ns.gluroo.com/api/v1/entries.json?token=2098657e-e58a-432d-89af-f4f59f8bb44b&find[date][$gt]=1730637775000&count=10000000"

# Hlavičky pro Supabase s autorizací
headers = {
    "apikey": SUPABASE_API_KEY,
    "Authorization": f"Bearer {SUPABASE_API_KEY}",
    "Content-Type": "application/json"
}

# Funkce pro získání nejnovějšího záznamu z Supabase tabulky
def get_latest_entry():
    url = f"{SUPABASE_URL}/rest/v1/{TABLE_NAME}?select=_id,date&order=date.desc&limit=1"
    response = requests.get(url, headers=headers)

    if response.status_code == 200 and response.json():
        return response.json()[0]  # Vrací nejnovější záznam
    else:
        print("Chyba při načítání nejnovějšího záznamu:", response.status_code)
        return None

# Funkce pro uložení nových hodnot do Supabase
def save_new_entries(entries):
    url = f"{SUPABASE_URL}/rest/v1/{TABLE_NAME}"

    for entry in entries:
        # Vypočítat sgv_mmol jako sgv / 18, pokud sgv existuje
        sgv_value = entry.get("sgv")
        sgv_mmol = round(sgv_value / 18, 2) if sgv_value is not None else None

        # Připravit JSON pro nový záznam včetně sgv_mmol
        data = {
            "_id": entry.get("_id"),
            "sgv": sgv_value,
            "sgv_mmol": sgv_mmol,
            "date": entry.get("date"),
            "dateString": entry.get("dateString"),
            "trend": entry.get("trend"),
            "direction": entry.get("direction"),
            "utcOffset": entry.get("utcOffset"),
            "sysTime": entry.get("sysTime")
        }

        # Poslat POST požadavek do Supabase
        response = requests.post(url, headers=headers, json=data)

        if response.status_code != 201:
            print(f"Chyba při ukládání záznamu {entry.get('_id')}: {response.status_code}, {response.text}")
        else:
            print(f"Záznam {entry.get('_id')} úspěšně uložen s hodnotou sgv_mmol: {sgv_mmol}")

# Funkce pro načtení dat z API
def fetch_data_from_api(latest_timestamp):
    # Dynamicky doplníme hodnotu `latest_timestamp` do URL API
    url = f"https://2098.ns.gluroo.com/api/v1/entries.json?token=2098657e-e58a-432d-89af-f4f59f8bb44b&find[date][$gt]={latest_timestamp}&count=10"

    response = requests.get(url)
    if response.status_code == 200:
        return response.json()  # Předpokládáme, že API vrací data ve formátu JSON
    else:
        print("Chyba při načítání dat z API:", response.status_code)
        return []

# Funkce pro filtrování a zpracování nových dat
def process_data(data, latest_timestamp, existing_ids):
    # Filtrování nových záznamů podle nejnovějšího času a kontrola duplicit podle `_id`
    new_entries = [entry for entry in data
                   if entry.get('_id') not in existing_ids and entry.get('date', 0) > latest_timestamp]

    # Seřazení nových záznamů od nejnovějších k nejstarším
    new_entries.sort(key=lambda x: x.get('date', 0), reverse=True)

    if new_entries:
        save_new_entries(new_entries)
        print(f"{len(new_entries)} nové hodnoty byly uloženy.")
    else:
        print("Žádné nové hodnoty k uložení.")

# Hlavní část skriptu
# Načtení nejnovějšího záznamu z databáze
latest_entry = get_latest_entry()
latest_timestamp = latest_entry['date'] if latest_entry else 0
existing_ids = {latest_entry['_id']} if latest_entry else set()

# Načtení dat z API s použitím latest_timestamp
data = fetch_data_from_api(latest_timestamp)
if data:
    process_data(data, latest_timestamp, existing_ids)
