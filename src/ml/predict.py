# src/ml/predict.py
import os
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from ml.prepare import FEATURES

def predict_next_day(df_ticker, model_path):
    """
    df_ticker: DataFrame of a single ticker features (from data/processed/features/...)
    model_path: path to saved regression model, e.g. "models/LightGBMReg.pkl"
    Returns predicted next-day price (float)
    """
    df = df_ticker.copy().dropna()
    if df.empty:
        raise ValueError("Input dataframe is empty after dropna()")

    if not all(f in df.columns for f in FEATURES):
        missing = [f for f in FEATURES if f not in df.columns]
        raise ValueError("Missing features: " + ", ".join(missing))

    X = df[FEATURES].values

    # Scale per-file (simple). For production, save scaler at train time and reuse.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_last = X_scaled[-1].reshape(1, -1)

    model = joblib.load(model_path)
    pred = model.predict(X_last)[0]
    return float(pred)
