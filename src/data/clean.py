import os
import pandas as pd

CLEAN_DIR = "data/processed/clean"
os.makedirs(CLEAN_DIR, exist_ok=True)

def clean_dataframe(df):
    df = df.copy()

    # --- 1) Chuẩn hóa cột Date ---
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # --- 2) Sử dụng Adj Close làm Close chuẩn ---
    if "Adj Close" in df.columns:
        df["Close"] = df["Adj Close"]

    # --- 3) Giữ các cột cần thiết ---
    cols_needed = ["Date", "Open", "High", "Low", "Close", "Volume", "Ticker"]
    cols_existing = [c for c in cols_needed if c in df.columns]
    df = df[cols_existing]

    # --- 4) Drop NA theo cột quan trọng ---
    must_have = ["Close", "Volume"]
    df = df.dropna(subset=must_have)

    # --- 5) Sort theo Ticker + Date ---
    sort_cols = [c for c in ["Ticker", "Date"] if c in df.columns]
    df = df.sort_values(sort_cols).reset_index(drop=True)

    return df

def save_clean(df, ticker):
    path = f"{CLEAN_DIR}/{ticker}.csv"
    df.to_csv(path, index=False)
    print(f"[OK] Saved clean → {path}")
    return df
