# src/data/clean.py
import os
import pandas as pd

CLEAN_DIR = "data/processed/clean"
os.makedirs(CLEAN_DIR, exist_ok=True)

def clean_dataframe(df):
    df = df.copy()

    # --- 1) Chuẩn hóa cột Date ---
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # --- 2) Giữ đúng các cột cần thiết (chỉ chọn cột có tồn tại) ---
    cols_needed = [
        "Date", "Open", "High", "Low", "Close",
        "Adj Close", "Volume", "Ticker"
    ]
    cols_existing = [c for c in cols_needed if c in df.columns]
    df = df[cols_existing]

    # --- 3) Drop NA theo cột quan trọng (nếu có) ---
    must_have = []
    if "Close" in df.columns:
        must_have.append("Close")
    if "Volume" in df.columns:
        must_have.append("Volume")

    if must_have:
        df = df.dropna(subset=must_have)

    # --- 4) Sort sạch sẽ ---
    sort_cols = [c for c in ["Ticker", "Date"] if c in df.columns]
    df = df.sort_values(sort_cols).reset_index(drop=True)

    return df


def save_clean(df, ticker):
    path = f"{CLEAN_DIR}/{ticker}.csv"
    df.to_csv(path, index=False)
    print(f"[OK] Saved clean → {path}")
    return df
