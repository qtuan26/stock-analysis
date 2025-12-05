# src/pipeline.py
from datetime import datetime
from fetch import fetch_data
from clean import clean_dataframe, save_clean
from features import add_indicators, save_features

TICKERS = ["AAPL", "AMZN", "GOOG", "MSFT", "TSLA"]

def run_pipeline(tickers=TICKERS):
    for ticker in tickers:
        # 1) RAW
        df_raw = fetch_data(ticker)
        if df_raw is None:
            continue

        # 2) CLEAN
        df_clean = clean_dataframe(df_raw)
        df_clean = save_clean(df_clean, ticker)

        # 3) FEATURES
        df_feat = add_indicators(df_clean)
        save_features(df_feat, ticker)

    print(f"\n[DONE] Pipeline completed at {datetime.now()}\n")

if __name__ == "__main__":
    run_pipeline()
