# src/ml/load_all_features.py
import os
import pandas as pd

FEATURE_DIR = os.path.join("data", "processed", "features")

def load_all_feature_files():
    dfs = []
    if not os.path.exists(FEATURE_DIR):
        raise ValueError(f"Feature dir not found: {FEATURE_DIR}")
    for file in os.listdir(FEATURE_DIR):
        if file.endswith("_features.csv") or file.endswith("_features.csv".lower()):
            path = os.path.join(FEATURE_DIR, file)
            df = pd.read_csv(path)
            # ensure Ticker column exists (if not, infer from filename)
            if "Ticker" not in df.columns:
                df["Ticker"] = os.path.basename(file).split("_")[0]
            dfs.append(df)
    if not dfs:
        raise ValueError("No feature files found in " + FEATURE_DIR)
    df_all = pd.concat(dfs, ignore_index=True)
    if "Date" in df_all.columns:
        df_all["Date"] = pd.to_datetime(df_all["Date"], errors="coerce")
        df_all = df_all.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    return df_all
