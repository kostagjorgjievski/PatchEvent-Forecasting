# src/preprocessing/patch_builder.py

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

PATCH_LEN = 30
TARGET_LEN = 15
FEATURES = ["sales", "onpromotion", "dcoilwtico", "transactions"]
OUTPUT_FILE = Path("data/processed/patch_dataset.parquet")

def extract_patches(df: pd.DataFrame):
    all_patches = []
    
    grouped = df.groupby(["store_nbr", "family"])
    
    for (store, fam), group in tqdm(grouped, desc="Building patches"):
        group = group.sort_values("date").reset_index(drop=True)
        values = group[FEATURES].values
        sales = group["sales"].values
        events = group["description"].values
        dates = group["date"].values
        
        max_idx = len(group) - PATCH_LEN - TARGET_LEN + 1
        
        for i in range(max_idx):
            x_patch = values[i:i+PATCH_LEN]
            y_target = sales[i+PATCH_LEN : i+PATCH_LEN+TARGET_LEN]
            event_slice = events[i:i+PATCH_LEN]
            window_events = " | ".join(sorted(set(e for e in event_slice if isinstance(e, str) and e.strip())))

            patch = {
                "store_nbr": store,
                "family": fam,
                "x_patch": x_patch.tolist(),
                "y_target": y_target.tolist(),
                "event_text": window_events,
                "start_date": dates[i],
                "end_date": dates[i+PATCH_LEN-1],
            }
            all_patches.append(patch)
    
    return all_patches

def main():
    df = pd.read_csv("data/interim/merged.csv", parse_dates=["date"])
    patches = extract_patches(df)
    df_out = pd.DataFrame(patches)
    df_out.to_parquet(OUTPUT_FILE, index=False)
    print(f"Saved {len(df_out)} patches to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
