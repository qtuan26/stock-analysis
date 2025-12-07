# src/data/fetch.py
import os
import yfinance as yf
import pandas as pd

RAW_DIR = "data/raw"
os.makedirs(RAW_DIR, exist_ok=True)

def fetch_data(ticker, period="5y"):
    print(f"[FETCH] {ticker}")
    try:
        df = yf.download(ticker, period=period, progress=False, auto_adjust=False)

        if df.empty:
            print(f"[SKIP] No data for {ticker}")
            return None

        df = df.reset_index()

        # --- FIX: Flatten MultiIndex columns ---
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [
                col[0] if col[1] == '' else f"{col[0]}_{col[1]}"
                for col in df.columns
            ]

        # --- FIX: Remove ticker suffix ---
        df.columns = [c.replace(f"_{ticker}", "") for c in df.columns]

        # add ticker col
        df["Ticker"] = ticker

        raw_path = f"{RAW_DIR}/{ticker}_raw.csv"
        df.to_csv(raw_path, index=False)
        print(f"[OK] Saved raw → {raw_path}")
        return df
    except Exception as e:
        print(f"[ERROR] Failed {ticker}: {e}")
        return None
