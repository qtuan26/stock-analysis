# src/ml/save_load.py
import joblib
import os

def save_model(model, path):
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, f"models/{path}")

def load_model(path):
    return joblib.load(f"models/{path}")
