# Version: V0.51
import os, json, pickle
import pandas as pd

def save_model(model, path: str):
    with open(path, "wb") as f:
        pickle.dump(model, f)

def load_model(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)

def save_schema(schema: dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(schema, f, ensure_ascii=False, indent=2)

def load_schema(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def align_features(df_feat: pd.DataFrame, schema: dict) -> pd.DataFrame:
    cols = schema["feature_names"]
    X = df_feat.reindex(columns=cols)
    return X.fillna(method="ffill").fillna(method="bfill")
