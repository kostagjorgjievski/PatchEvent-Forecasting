# src/preprocessing/merge_sources.py

import pandas as pd
from pathlib import Path

RAW_DATA_DIR = Path("data/raw/")
OUTPUT_PATH = Path("data/interim/merged.csv")

def load_and_merge():
    # Load datasets
    train = pd.read_csv(RAW_DATA_DIR / "train.csv", parse_dates=["date"])
    oil = pd.read_csv(RAW_DATA_DIR / "oil.csv", parse_dates=["date"])
    holidays = pd.read_csv(RAW_DATA_DIR / "holidays_events.csv", parse_dates=["date"])
    trans = pd.read_csv(RAW_DATA_DIR / "transactions.csv", parse_dates=["date"])

    # Normalize and clean holidays
    holidays = holidays[holidays['locale'] != 'National']  
    holidays['description'] = holidays['description'].str.lower().str.strip()

    translation_map = {
        "navidad": "christmas",
        "año nuevo": "new year",
        "carnaval": "carnival",
        "puente": "bridge holiday",
        "terremoto": "earthquake",
        # add more as well
    }

    holidays['description'] = holidays['description'].replace(translation_map, regex=True)

    # Group events by date and aggregate
    event_text = holidays.groupby('date')['description'].apply(lambda x: ' | '.join(sorted(set(x)))).reset_index()

    # Merge sources (left join by date)
    df = train.merge(oil, how="left", on="date")
    df = df.merge(event_text, how="left", on="date")
    df = df.merge(trans, how="left", on=["date", "store_nbr"])

    df['transactions'] = df['transactions'].fillna(0)
    df['dcoilwtico'] = df['dcoilwtico'].ffill()
    df['description'] = df['description'].fillna("")

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Merged dataset saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    load_and_merge()
