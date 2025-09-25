# Version: V0.51
import os
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from config import *
from data.fetcher import get_ohlcv
from data.features import compute_features
from data.labeling import label_data
from model.utils import save_model, save_schema

def compute_class_weights(y_int: np.ndarray):
    unique, counts = np.unique(y_int, return_counts=True)
    N = len(y_int); K = len(unique)
    base = {cls: N / (K * cnt) for cls, cnt in zip(unique, counts)}
    return np.array([base[i] for i in y_int], dtype=float)

def train_for_horizon(H: int):
    print(f"\n=== Entraînement H={H} ===")
    os.makedirs(MODEL_DIR, exist_ok=True)

    df = get_ohlcv(SYMBOL, TIMEFRAME, RETRAIN_LOOKBACK_BARS)
    if df is None or len(df) < MIN_BARS:
        print("Données insuffisantes pour entraîner."); return

    df_feat = compute_features(df)
    if df_feat is None or len(df_feat) < MIN_BARS:
        print("Features insuffisantes pour entraîner."); return

    X, y_text, feature_cols = label_data(df_feat, H, LABEL_SEUILS)
    y_text = y_text.replace({"NEUTRE": "RANGE"})
    if X.empty:
        print("Impossible de labelliser (X vide)."); return

    split = int(len(X) * 0.8)
    X_train, X_valid = X.iloc[:split], X.iloc[split:]
    y_train_text, y_valid_text = y_text.iloc[:split], y_text.iloc[split:]

    le_train = LabelEncoder()
    y_train = le_train.fit_transform(y_train_text)
    classes_train = le_train.classes_.tolist()
    n_classes_train = len(classes_train)

    mask_valid = y_valid_text.isin(classes_train)
    if not mask_valid.all():
        dropped = int((~mask_valid).sum())
        print(f"⚠ {dropped} échantillon(s) de validation ignoré(s) (classe absente du TRAIN).")
    X_valid = X_valid[mask_valid]
    y_valid_text = y_valid_text[mask_valid]
    y_valid = le_train.transform(y_valid_text) if len(y_valid_text) > 0 else np.array([], dtype=int)

    sample_weight = compute_class_weights(y_train)

    params = dict(XGB_PARAMS)
    if n_classes_train == 2:
        params["objective"] = "binary:logistic"
        params.pop("num_class", None)
    else:
        params["objective"] = "multi:softprob"
        params["num_class"] = n_classes_train

    model = XGBClassifier(**params)
    model.fit(X_train, y_train, sample_weight=sample_weight)

    if y_valid.size > 0:
        y_pred = model.predict(X_valid)
        metrics = {
            "accuracy": float(accuracy_score(y_valid, y_pred)),
            "f1_macro": float(f1_score(y_valid, y_pred, average="macro")),
            "confusion_matrix": confusion_matrix(y_valid, y_pred).tolist(),
            "n_classes_train": n_classes_train,
            "classes_train": classes_train,
        }
    else:
        metrics = {
            "accuracy": None, "f1_macro": None, "confusion_matrix": None,
            "n_classes_train": n_classes_train, "classes_train": classes_train,
        }
    print(f"=== Évaluation H={H} ===")
    print(metrics)

    model_path = os.path.join(MODEL_DIR, f"xgb_H{H}.pkl")
    schema_path = os.path.join(MODEL_DIR, f"schema_H{H}.json")
    save_model(model, model_path)
    save_schema({"feature_names": feature_cols, "classes_": classes_train}, schema_path)
    print(f"✅ Modèle sauvegardé: {model_path}")
    print(f"✅ Schéma sauvegardé: {schema_path}")

def main():
    for H in HORIZONS:
        train_for_horizon(H)

if __name__ == "__main__":
    main()
