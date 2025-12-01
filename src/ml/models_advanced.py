# src/ml/models_advanced.py
import numpy as np
from xgboost import XGBRegressor
import lightgbm as lgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# ==================== XGBOOST ====================
def train_xgb(X_train, y_train):
    model = XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror"
    )
    model.fit(X_train, y_train)
    return model


# ==================== LIGHTGBM ====================
def train_lgb(X_train, y_train):
    model = lgb.LGBMRegressor(
        n_estimators=500,
        num_leaves=31,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8
    )
    model.fit(X_train, y_train)
    return model


# ==================== LSTM ====================
def reshape_for_lstm(X, timesteps=20):
    X_lstm = []
    for i in range(timesteps, len(X)):
        X_lstm.append(X[i-timesteps:i])
    return np.array(X_lstm)

def build_lstm(input_shape):
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(32))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")
    return model
