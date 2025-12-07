# src/data/features.py
import os
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator, EMAIndicator
from ta.volatility import BollingerBands

FEATURE_DIR  = "data/processed/features"

def add_indicators(df):
    """
    Thêm các chỉ báo kỹ thuật: RSI, SMA, EMA, Bollinger Bands
    """
    if df.empty or len(df) < 20:
        print("[SKIP] Not enough data for indicators")
        return None

    df = df.copy()
    close = df['Close'].squeeze()
    
    df['rsi_14'] = RSIIndicator(close=close, window=14).rsi()
    df['sma_20'] = SMAIndicator(close=close, window=20).sma_indicator()
    df['ema_12'] = EMAIndicator(close=close, window=12).ema_indicator()
    bb = BollingerBands(close=close, window=20)
    df['bb_h'] = bb.bollinger_hband()
    df['bb_l'] = bb.bollinger_lband()

    df = df.dropna().reset_index(drop=True)
    return df

def save_features(df, ticker):
    """
    Lưu file features per ticker
    """
    if df is None or df.empty:
        return None
    out_file = os.path.join(FEATURE_DIR, f"{ticker}_features.csv")
    df.to_csv(out_file, index=False)
    print(f"[OK] Saved features → {out_file}")
    return df
