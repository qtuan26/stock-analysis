# task4_ml_pipeline_fixed.py
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve, auc, average_precision_score
)
from sklearn.model_selection import ParameterGrid
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

import lightgbm as lgb
import xgboost as xgb
import shap

# ----------------------------
# Utility functions
# ----------------------------
def ensure_dirs(dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def max_drawdown(cum_returns):
    roll_max = cum_returns.cummax()
    drawdown = cum_returns / roll_max - 1.0
    return drawdown.min()

def annualized_sharpe(daily_returns, days_per_year=252):
    mean = daily_returns.mean() * days_per_year
    std = daily_returns.std() * np.sqrt(days_per_year)
    if std == 0:
        return np.nan
    return mean / std

def save_fig(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)

# Safe XGBoost predict to handle different xgb versions
def safe_xgb_predict(booster, dmatrix):
    """
    Use best_ntree_limit if available; fallback to best_iteration; otherwise predict normally.
    This avoids AttributeError on some xgboost versions.
    """
    ntree_limit = None
    if hasattr(booster, "best_ntree_limit") and getattr(booster, "best_ntree_limit") is not None:
        ntree_limit = booster.best_ntree_limit
    elif hasattr(booster, "best_iteration") and getattr(booster, "best_iteration") is not None:
        ntree_limit = getattr(booster, "best_iteration")
    try:
        if ntree_limit is not None:
            return booster.predict(dmatrix, ntree_limit=ntree_limit)
        else:
            return booster.predict(dmatrix)
    except TypeError:
        # some versions expect different signature - fallback
        return booster.predict(dmatrix)

# ----------------------------
# Load features
# ----------------------------
DATA_FILE = "results/features_engineered.csv" if os.path.exists("results/features_engineered.csv") else "features_engineered.csv"
assert os.path.exists(DATA_FILE), f"{DATA_FILE} not found. Run Task 3 first."

df = pd.read_csv(DATA_FILE, parse_dates=["date"])
df = df.sort_values(["date", "symbol"]).reset_index(drop=True)

# Drop rows where label is NaN (just in case)
df = df.dropna(subset=["label_up_down", "target_return"])

# ----------------------------
# Config & outputs
# ----------------------------
OUT_DIRS = ["models", "plots", "results"]
ensure_dirs(OUT_DIRS)

MODEL_PATH_LGB = os.path.join("models", "lgb_model.txt")
MODEL_PATH_XGB = os.path.join("models", "xgb_model.json")
MODEL_PATH_RF = os.path.join("models", "rf_model.pkl")
MODEL_PATH_GB = os.path.join("models", "gb_model.pkl")
MODEL_PATH_LR = os.path.join("models", "lr_model.pkl")
SCALER_PATH = os.path.join("models", "scaler.pkl")
RESULTS_CSV = os.path.join("results", "ml_results_summary.csv")

# ----------------------------
# Feature matrix
# ----------------------------
exclude_cols = {"date", "symbol", "label_up_down", "target_return"}
feature_cols = [c for c in df.columns if c not in exclude_cols]
X_all = df[feature_cols].copy()
y_all = df["label_up_down"].astype(int).copy()

# ----------------------------
# Walk-forward split function
# ----------------------------
def get_time_splits(df_dates, n_splits=5, min_train_days=252):
    """
    Build expanding-window splits by date.
    Returns list of (train_idx, val_idx) index arrays (positions in df)
    """
    dates = np.sort(df_dates.dt.normalize().unique())
    if len(dates) < n_splits + 1:
        n_splits = max(1, len(dates) - 1)
    split_points = np.linspace(0, len(dates) - 1, n_splits + 1, dtype=int)[1:]
    splits = []
    for pt in split_points:
        val_start_date = dates[pt]
        # train: dates strictly less than val_start_date
        train_mask = df_dates.dt.normalize() < val_start_date
        val_mask = df_dates.dt.normalize() == val_start_date
        if train_mask.sum() < min_train_days:
            continue
        train_idx = np.where(train_mask)[0]
        val_idx = np.where(val_mask)[0]
        if len(val_idx) == 0:
            continue
        splits.append((train_idx, val_idx))
    return splits

# Build splits
splits = get_time_splits(df["date"], n_splits=6)
if len(splits) == 0:
    # fallback: 80/20 by row index
    n = len(df)
    split_idx = int(n * 0.8)
    splits = [(np.arange(0, split_idx), np.arange(split_idx, n))]

print(f"[INFO] Created {len(splits)} time splits for walk-forward training")

# ----------------------------
# Training loop: LightGBM + XGBoost
# ----------------------------
lgb_params = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "verbosity": -1,
    "seed": 42,
    "n_jobs": -1,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "min_data_in_leaf": 20,
}

xgb_params = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "use_label_encoder": False,
    "seed": 42,
    "eta": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
}

rf_params = {
    "n_estimators": 100,
    "max_depth": 10,
    "min_samples_split": 10,
    "min_samples_leaf": 5,
    "random_state": 42,
    "n_jobs": -1,
    "class_weight": "balanced",
}

gb_params = {
    "n_estimators": 100,
    "learning_rate": 0.05,
    "max_depth": 5,
    "min_samples_split": 10,
    "min_samples_leaf": 5,
    "subsample": 0.8,
    "random_state": 42,
}

lr_params = {
    "max_iter": 1000,
    "random_state": 42,
    "n_jobs": -1,
    "class_weight": "balanced",
}

oof_preds_lgb = np.zeros(len(df))
oof_preds_xgb = np.zeros(len(df))
oof_preds_rf = np.zeros(len(df))
oof_preds_gb = np.zeros(len(df))
oof_preds_lr = np.zeros(len(df))
fold_metrics = []

for fold, (train_idx, val_idx) in enumerate(splits, 1):
    print(f"\n=== Fold {fold} ===")
    X_train = X_all.iloc[train_idx].copy()
    X_val   = X_all.iloc[val_idx].copy()
    y_train = y_all.iloc[train_idx].copy()
    y_val   = y_all.iloc[val_idx].copy()
    
    # Scale features (fit on train only)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)
    
    # Save scaler of last fold (we'll use last scaler as final)
    joblib.dump(scaler, SCALER_PATH)
    
    # handle class imbalance for LightGBM: compute scale_pos_weight
    pos = (y_train == 1).sum()
    neg = (y_train == 0).sum()
    scale_pos_weight = neg / max(1, pos)
    lgb_params_fold = lgb_params.copy()
    lgb_params_fold["scale_pos_weight"] = scale_pos_weight
    
    # LightGBM dataset
    lgb_train = lgb.Dataset(X_train_s, label=y_train)
    lgb_val = lgb.Dataset(X_val_s, label=y_val, reference=lgb_train)
    
    print("Training LightGBM...")
    # --- FIX: use callbacks for early stopping and logging (compatible with lgb v4+)
    lgb_model = lgb.train(
        lgb_params_fold,
        lgb_train,
        num_boost_round=2000,
        valid_sets=[lgb_train, lgb_val],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100),
            lgb.log_evaluation(-1)  # disable per-iteration printing; change to e.g. 50 to log
        ],
    )
    
    # Predict val (use best_iteration if available)
    if hasattr(lgb_model, "best_iteration") and lgb_model.best_iteration is not None:
        y_val_proba_lgb = lgb_model.predict(X_val_s, num_iteration=lgb_model.best_iteration)
    else:
        y_val_proba_lgb = lgb_model.predict(X_val_s)
    oof_preds_lgb[val_idx] = y_val_proba_lgb
    
    # XGBoost train
    xgb_dtrain = xgb.DMatrix(X_train_s, label=y_train)
    xgb_dval = xgb.DMatrix(X_val_s, label=y_val)
    xgb_watch = [(xgb_dtrain, "train"), (xgb_dval, "eval")]
    print("Training XGBoost...")
    xgb_model = xgb.train(
        params=xgb_params,
        dtrain=xgb_dtrain,
        num_boost_round=1000,
        evals=xgb_watch,
        early_stopping_rounds=100,
        verbose_eval=False
    )
    # Safe predict wrapper to avoid AttributeError across xgb versions
    y_val_proba_xgb = safe_xgb_predict(xgb_model, xgb_dval)
    oof_preds_xgb[val_idx] = y_val_proba_xgb
    
    # Random Forest
    print("Training Random Forest...")
    rf_model = RandomForestClassifier(**rf_params)
    rf_model.fit(X_train_s, y_train)
    y_val_proba_rf = rf_model.predict_proba(X_val_s)[:, 1]
    oof_preds_rf[val_idx] = y_val_proba_rf
    
    # Gradient Boosting
    print("Training Gradient Boosting...")
    gb_model = GradientBoostingClassifier(**gb_params)
    gb_model.fit(X_train_s, y_train)
    y_val_proba_gb = gb_model.predict_proba(X_val_s)[:, 1]
    oof_preds_gb[val_idx] = y_val_proba_gb
    
    # Logistic Regression
    print("Training Logistic Regression...")
    lr_model = LogisticRegression(**lr_params)
    lr_model.fit(X_train_s, y_train)
    y_val_proba_lr = lr_model.predict_proba(X_val_s)[:, 1]
    oof_preds_lr[val_idx] = y_val_proba_lr
    
    # Evaluate fold
    def classify_stats(y_true, y_proba, thr=0.5):
        y_pred = (y_proba >= thr).astype(int)
        return {
            "acc": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "auc": roc_auc_score(y_true, y_proba)
        }
    
    stats_lgb = classify_stats(y_val, y_val_proba_lgb)
    stats_xgb = classify_stats(y_val, y_val_proba_xgb)
    stats_rf = classify_stats(y_val, y_val_proba_rf)
    stats_gb = classify_stats(y_val, y_val_proba_gb)
    stats_lr = classify_stats(y_val, y_val_proba_lr)
    print("LGB Fold metrics:", stats_lgb)
    print("XGB Fold metrics:", stats_xgb)
    print("RF Fold metrics:", stats_rf)
    print("GB Fold metrics:", stats_gb)
    print("LR Fold metrics:", stats_lr)
    
    fold_metrics.append({
        "fold": fold,
        "lgb_acc": stats_lgb["acc"],
        "lgb_f1": stats_lgb["f1"],
        "lgb_auc": stats_lgb["auc"],
        "xgb_acc": stats_xgb["acc"],
        "xgb_f1": stats_xgb["f1"],
        "xgb_auc": stats_xgb["auc"],
        "rf_acc": stats_rf["acc"],
        "rf_f1": stats_rf["f1"],
        "rf_auc": stats_rf["auc"],
        "gb_acc": stats_gb["acc"],
        "gb_f1": stats_gb["f1"],
        "gb_auc": stats_gb["auc"],
        "lr_acc": stats_lr["acc"],
        "lr_f1": stats_lr["f1"],
        "lr_auc": stats_lr["auc"],
    })
    
    # Save last fold models to disk (will overwrite as we go; final saved will be last fold trainer)
    lgb_model.save_model(MODEL_PATH_LGB)
    xgb_model.save_model(MODEL_PATH_XGB)
    joblib.dump(rf_model, MODEL_PATH_RF)
    joblib.dump(gb_model, MODEL_PATH_GB)
    joblib.dump(lr_model, MODEL_PATH_LR)

# ----------------------------
# Final evaluation on OOF (out-of-fold)
# ----------------------------
def evaluate_oof(y_true, proba):
    auc_score = roc_auc_score(y_true, proba)
    ap_score = average_precision_score(y_true, proba)
    # best threshold by F1 (simple search)
    thrs = np.linspace(0.1, 0.9, 81)
    best_thr, best_f1 = 0.5, 0
    for t in thrs:
        f1v = f1_score(y_true, (proba >= t).astype(int), zero_division=0)
        if f1v > best_f1:
            best_f1 = f1v
            best_thr = t
    y_pred_best = (proba >= best_thr).astype(int)
    cr = classification_report(y_true, y_pred_best, digits=4)
    cm = confusion_matrix(y_true, y_pred_best)
    return {
        "auc": auc_score,
        "average_precision": ap_score,
        "best_threshold": best_thr,
        "best_f1": best_f1,
        "classification_report": cr,
        "confusion_matrix": cm
    }

print("\n=== OOF Evaluation ===")
res_lgb = evaluate_oof(y_all, oof_preds_lgb)
res_xgb = evaluate_oof(y_all, oof_preds_xgb)
res_rf = evaluate_oof(y_all, oof_preds_rf)
res_gb = evaluate_oof(y_all, oof_preds_gb)
res_lr = evaluate_oof(y_all, oof_preds_lr)

print("\nLightGBM OOF: AUC=%.4f AP=%.4f best_thr=%.3f best_f1=%.4f" %
      (res_lgb["auc"], res_lgb["average_precision"], res_lgb["best_threshold"], res_lgb["best_f1"]))
print(res_lgb["classification_report"])
print("Confusion matrix:\n", res_lgb["confusion_matrix"])

print("\nXGBoost OOF: AUC=%.4f AP=%.4f best_thr=%.3f best_f1=%.4f" %
      (res_xgb["auc"], res_xgb["average_precision"], res_xgb["best_threshold"], res_xgb["best_f1"]))
print(res_xgb["classification_report"])
print("Confusion matrix:\n", res_xgb["confusion_matrix"])

print("\nRandom Forest OOF: AUC=%.4f AP=%.4f best_thr=%.3f best_f1=%.4f" %
      (res_rf["auc"], res_rf["average_precision"], res_rf["best_threshold"], res_rf["best_f1"]))
print(res_rf["classification_report"])
print("Confusion matrix:\n", res_rf["confusion_matrix"])

print("\nGradient Boosting OOF: AUC=%.4f AP=%.4f best_thr=%.3f best_f1=%.4f" %
      (res_gb["auc"], res_gb["average_precision"], res_gb["best_threshold"], res_gb["best_f1"]))
print(res_gb["classification_report"])
print("Confusion matrix:\n", res_gb["confusion_matrix"])

print("\nLogistic Regression OOF: AUC=%.4f AP=%.4f best_thr=%.3f best_f1=%.4f" %
      (res_lr["auc"], res_lr["average_precision"], res_lr["best_threshold"], res_lr["best_f1"]))
print(res_lr["classification_report"])
print("Confusion matrix:\n", res_lr["confusion_matrix"])

# Save OOF fold metrics
pd.DataFrame(fold_metrics).to_csv(os.path.join("results", "fold_metrics.csv"), index=False)

# ----------------------------
# Economic evaluation / backtest (simple)
# ----------------------------
df_eval = df.copy()
df_eval["lgb_proba"] = oof_preds_lgb
df_eval["xgb_proba"] = oof_preds_xgb
df_eval["rf_proba"] = oof_preds_rf
df_eval["gb_proba"] = oof_preds_gb
df_eval["lr_proba"] = oof_preds_lr

thr_lgb = res_lgb["best_threshold"]
thr_xgb = res_xgb["best_threshold"]
thr_rf = res_rf["best_threshold"]
thr_gb = res_gb["best_threshold"]
thr_lr = res_lr["best_threshold"]

df_eval["lgb_signal"] = (df_eval["lgb_proba"] >= thr_lgb).astype(int)
df_eval["xgb_signal"] = (df_eval["xgb_proba"] >= thr_xgb).astype(int)
df_eval["rf_signal"] = (df_eval["rf_proba"] >= thr_rf).astype(int)
df_eval["gb_signal"] = (df_eval["gb_proba"] >= thr_gb).astype(int)
df_eval["lr_signal"] = (df_eval["lr_proba"] >= thr_lr).astype(int)

df_eval["next_ret"] = df_eval["target_return"] / 100.0  # convert to decimal
TRANSACTION_COST = 0.0005  # 5 bps per trade (adjustable)

df_eval = df_eval.sort_values(["symbol", "date"]).reset_index(drop=True)
df_eval["lgb_pos"] = df_eval.groupby("symbol")["lgb_signal"].shift(0).fillna(0)
df_eval["lgb_pos_prev"] = df_eval.groupby("symbol")["lgb_pos"].shift(1).fillna(0)
df_eval["lgb_trade"] = (df_eval["lgb_pos"] != df_eval["lgb_pos_prev"]).astype(int)

df_eval["xgb_pos"] = df_eval.groupby("symbol")["xgb_signal"].shift(0).fillna(0)
df_eval["xgb_pos_prev"] = df_eval.groupby("symbol")["xgb_pos"].shift(1).fillna(0)
df_eval["xgb_trade"] = (df_eval["xgb_pos"] != df_eval["xgb_pos_prev"]).astype(int)

df_eval["rf_pos"] = df_eval.groupby("symbol")["rf_signal"].shift(0).fillna(0)
df_eval["rf_pos_prev"] = df_eval.groupby("symbol")["rf_pos"].shift(1).fillna(0)
df_eval["rf_trade"] = (df_eval["rf_pos"] != df_eval["rf_pos_prev"]).astype(int)

df_eval["gb_pos"] = df_eval.groupby("symbol")["gb_signal"].shift(0).fillna(0)
df_eval["gb_pos_prev"] = df_eval.groupby("symbol")["gb_pos"].shift(1).fillna(0)
df_eval["gb_trade"] = (df_eval["gb_pos"] != df_eval["gb_pos_prev"]).astype(int)

df_eval["lr_pos"] = df_eval.groupby("symbol")["lr_signal"].shift(0).fillna(0)
df_eval["lr_pos_prev"] = df_eval.groupby("symbol")["lr_pos"].shift(1).fillna(0)
df_eval["lr_trade"] = (df_eval["lr_pos"] != df_eval["lr_pos_prev"]).astype(int)

df_eval["lgb_strategy_ret"] = df_eval["lgb_pos"] * df_eval["next_ret"] - df_eval["lgb_trade"] * TRANSACTION_COST
df_eval["xgb_strategy_ret"] = df_eval["xgb_pos"] * df_eval["next_ret"] - df_eval["xgb_trade"] * TRANSACTION_COST
df_eval["rf_strategy_ret"] = df_eval["rf_pos"] * df_eval["next_ret"] - df_eval["rf_trade"] * TRANSACTION_COST
df_eval["gb_strategy_ret"] = df_eval["gb_pos"] * df_eval["next_ret"] - df_eval["gb_trade"] * TRANSACTION_COST
df_eval["lr_strategy_ret"] = df_eval["lr_pos"] * df_eval["next_ret"] - df_eval["lr_trade"] * TRANSACTION_COST

daily = df_eval.groupby("date").agg(
    lgb_strat_ret = ("lgb_strategy_ret", "mean"),
    xgb_strat_ret = ("xgb_strategy_ret", "mean"),
    rf_strat_ret = ("rf_strategy_ret", "mean"),
    gb_strat_ret = ("gb_strategy_ret", "mean"),
    lr_strat_ret = ("lr_strategy_ret", "mean"),
).sort_index()

for name in ["lgb_strat_ret", "xgb_strat_ret", "rf_strat_ret", "gb_strat_ret", "lr_strat_ret"]:
    daily[f"{name}_cum"] = (1 + daily[name].fillna(0)).cumprod()

for model_prefix in ["lgb", "xgb", "rf", "gb", "lr"]:
    strat = daily[f"{model_prefix}_strat_ret"].fillna(0)
    cum = daily[f"{model_prefix}_strat_ret_cum"]
    ann_sharpe = annualized_sharpe(strat)
    mdd = max_drawdown(cum)
    total_return = cum.iloc[-1] - 1.0
    print(f"\n[{model_prefix.upper()}] Total return = {total_return:.2%}, Sharpe = {ann_sharpe:.3f}, Max drawdown = {mdd:.2%}")

# ----------------------------
# Plots: ROC & confusion for LGB (OOF)
# ----------------------------
fpr, tpr, _ = roc_curve(y_all, oof_preds_lgb)
roc_auc = auc(fpr, tpr)
fig = plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"LGB OOF ROC (AUC = {roc_auc:.3f})")
plt.plot([0,1],[0,1],'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC - LightGBM (OOF)")
plt.legend()
save_fig(fig, os.path.join("plots", "roc_lgb_oof.png"))

best_thr = res_lgb["best_threshold"]
y_pred_best = (oof_preds_lgb >= best_thr).astype(int)
cm = confusion_matrix(y_all, y_pred_best)
fig = plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title(f"LGB Confusion Matrix (thr={best_thr:.2f})")
plt.xlabel("Predicted")
plt.ylabel("Actual")
save_fig(fig, os.path.join("plots", "confusion_lgb.png"))

# Feature importance (LGB)
lgb_final = lgb.Booster(model_file=MODEL_PATH_LGB)
imp = pd.DataFrame({
    "feature": feature_cols,
    "importance": lgb_final.feature_importance(importance_type="gain")
}).sort_values("importance", ascending=False)
imp.to_csv(os.path.join("results", "lgb_feature_importance.csv"), index=False)

fig = plt.figure(figsize=(8,10))
topn = imp.head(30)
plt.barh(topn["feature"][::-1], topn["importance"][::-1])
plt.xlabel("Gain importance")
plt.title("Top feature importance - LightGBM")
save_fig(fig, os.path.join("plots", "lgb_feature_importance.png"))

# ----------------------------
# SHAP explainability (sample) - compute on SCALED data and use feature names
# ----------------------------
print("\n[INFO] Computing SHAP values on a sample (may take time)...")
explainer = shap.TreeExplainer(lgb_final)

# sample index
sample_idx = np.random.choice(len(X_all), min(5000, len(X_all)), replace=False)
X_sample = X_all.iloc[sample_idx]
# load scaler and transform
scaler = joblib.load(SCALER_PATH)
X_sample_s = scaler.transform(X_sample)

# compute shap values on scaled sample, and pass feature names to summary_plot
sv = explainer.shap_values(X_sample_s)
# shap API may return list for multiclass; choose index 1 if list
shap_values = sv[1] if isinstance(sv, list) else sv
fig = plt.figure(figsize=(8,6))
# shap.summary_plot accepts numpy array + feature_names
shap.summary_plot(shap_values, X_sample_s, feature_names=X_sample.columns.tolist(), show=False)
save_fig(fig, os.path.join("plots", "shap_summary.png"))

# SHAP bar plot - average impact of each feature
fig = plt.figure(figsize=(10,6))
shap.summary_plot(shap_values, X_sample_s, plot_type="bar", feature_names=X_sample.columns.tolist(), show=False)
save_fig(fig, os.path.join("plots", "shap_bar_importance.png"))

# SHAP force plot - explain a single prediction
fig = plt.figure(figsize=(12,3))
shap.force_plot(explainer.expected_value, shap_values[0:1], X_sample_s[0:1], feature_names=X_sample.columns.tolist(), matplotlib=True, show=False)
save_fig(fig, os.path.join("plots", "shap_force_example.png"))

# Save final artifacts
joblib.dump(lgb_final, os.path.join("models", "lgb_final.pkl"))
summary = {
    "lgb_oof_auc": res_lgb["auc"],
    "lgb_oof_ap": res_lgb["average_precision"],
    "lgb_best_thr": res_lgb["best_threshold"],
    "lgb_best_f1": res_lgb["best_f1"],
    "xgb_oof_auc": res_xgb["auc"],
    "xgb_oof_ap": res_xgb["average_precision"],
    "xgb_best_thr": res_xgb["best_threshold"],
    "xgb_best_f1": res_xgb["best_f1"],
    "rf_oof_auc": res_rf["auc"],
    "rf_oof_ap": res_rf["average_precision"],
    "rf_best_thr": res_rf["best_threshold"],
    "rf_best_f1": res_rf["best_f1"],
    "gb_oof_auc": res_gb["auc"],
    "gb_oof_ap": res_gb["average_precision"],
    "gb_best_thr": res_gb["best_threshold"],
    "gb_best_f1": res_gb["best_f1"],
    "lr_oof_auc": res_lr["auc"],
    "lr_oof_ap": res_lr["average_precision"],
    "lr_best_thr": res_lr["best_threshold"],
    "lr_best_f1": res_lr["best_f1"],
}
pd.DataFrame([summary]).to_csv(RESULTS_CSV, index=False)

daily.to_csv(os.path.join("results", "daily_backtest.csv"))

print("\n✅ Task 4 completed. Artifacts saved in folders: models/, plots/, results/")
