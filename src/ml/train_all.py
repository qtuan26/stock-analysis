# src/ml/train_all.py
import os
import json
import joblib
from load_all_features import load_all_feature_files
from prepare import prepare_ml_data
from models_traditional import (
    train_linear_reg, train_rf_reg,
    train_logistic, train_rf_clf
)
from models_advanced import train_xgb, train_lgb
from evaluate import eval_reg, eval_clf
from save_load import save_model

OUTPUT_DIR = "models"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def train_all():
    print("[STEP 1] Load all feature files")
    df = load_all_feature_files()
    print(f"  -> rows: {len(df)}, tickers: {df['Ticker'].nunique()}")

    print("[STEP 2] Prepare data")
    Xtr_r, Xte_r, ytr_r, yte_r, Xtr_c, Xte_c, ytr_c, yte_c = prepare_ml_data(df)

    models = {}
    results = {"regression": {}, "classification": {}}

    print("[STEP 3] Train regression models")
    lin = train_linear_reg(Xtr_r, ytr_r)
    rf  = train_rf_reg(Xtr_r, ytr_r)
    xgb = train_xgb(Xtr_r, ytr_r)
    lgb = train_lgb(Xtr_r, ytr_r)

    models.update({
        "LinearReg": lin,
        "RandomForestReg": rf,
        "XGBoostReg": xgb,
        "LightGBMReg": lgb
    })

    # evaluate (each eval_reg returns dict {"rmse": value})
    results["regression"] = {
        "LinearReg": eval_reg(yte_r, lin.predict(Xte_r)),
        "RandomForestReg": eval_reg(yte_r, rf.predict(Xte_r)),
        "XGBoostReg": eval_reg(yte_r, xgb.predict(Xte_r)),
        "LightGBMReg": eval_reg(yte_r, lgb.predict(Xte_r)),
    }

    # choose best regression by rmse
    best_reg = min(results["regression"].keys(), key=lambda k: results["regression"][k]["rmse"])
    best_reg_model = models[best_reg]
    print(f"  -> Best regression model: {best_reg} (RMSE={results['regression'][best_reg]['rmse']:.6f})")

    print("[STEP 4] Train classification models")
    log = train_logistic(Xtr_c, ytr_c)
    rfc = train_rf_clf(Xtr_c, ytr_c)

    models.update({
        "LogisticClf": log,
        "RandomForestClf": rfc
    })

    results["classification"] = {
        "LogisticClf": eval_clf(yte_c, log.predict(Xte_c)),
        "RandomForestClf": eval_clf(yte_c, rfc.predict(Xte_c)),
    }

    # choose best classification by f1
    best_clf = max(results["classification"].keys(), key=lambda k: results["classification"][k]["f1"])
    best_clf_model = models[best_clf]
    print(f"  -> Best classification model: {best_clf} (F1={results['classification'][best_clf]['f1']:.6f})")

    print("[STEP 5] Save best models & report")
    save_model(best_reg_model, f"{best_reg}.pkl")
    save_model(best_clf_model, f"{best_clf}.pkl")

    # Save all models optionally
    # for name, m in models.items():
    #     save_model(m, f"all_{name}.pkl")

    # Save JSON report
    with open(os.path.join(OUTPUT_DIR, "model_report.json"), "w") as f:
        json.dump(results, f, indent=4)

    print("[DONE] Training complete. Models saved to", OUTPUT_DIR)
    return models, results, best_reg, best_clf

if __name__ == "__main__":
    train_all()
