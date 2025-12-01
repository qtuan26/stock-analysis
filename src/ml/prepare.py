# src/ml/prepare.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

FEATURES = [
    "Close", "Volume", "rsi_14", "sma_20",
    "ema_12", "bb_h", "bb_l"
]

def prepare_ml_data(df):
    df = df.dropna()

    # Tạo targets
    df["target_reg"] = df["Close"].shift(-1)
    df["target_clf"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

    df = df.dropna()

    X = df[FEATURES]
    y_reg = df["target_reg"]
    y_clf = df["target_clf"]

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X_scaled, y_reg, test_size=0.2, shuffle=False
    )

    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X_scaled, y_clf, test_size=0.2, shuffle=False
    )

    return X_train_r, X_test_r, y_train_r, y_test_r, \
           X_train_c, X_test_c, y_train_c, y_test_c
