# Version: V0.51
import pandas as pd
from collections import Counter
from config import LABELING_MODE, LABEL_SEUILS, QUANTILES, MIN_PER_CLASS

def future_return(close: pd.Series, horizon: int) -> pd.Series:
    return close.shift(-horizon) / close - 1.0

def _label_by_threshold(ret: float, seuils: dict) -> str:
    if ret >= seuils["up"]:
        return "UP"
    if ret <= seuils["down"]:
        return "DOWN"
    return "RANGE"

def _label_by_quantile(ret: float, q_low: float, q_high: float) -> str:
    if ret <= q_low:  return "DOWN"
    if ret >= q_high: return "UP"
    return "RANGE"

def label_data(df_feat: pd.DataFrame, horizon: int, _unused_seuils: dict):
    df = df_feat.copy()
    fut_ret = future_return(df["close"], horizon)

    exclude = {"time"}
    feature_cols = [c for c in df.columns if c not in exclude]
    X_all = df[feature_cols]

    valid_idx = fut_ret.dropna().index
    valid_idx = valid_idx[:-horizon]
    X = X_all.loc[valid_idx].reset_index(drop=True)
    fut = fut_ret.loc[valid_idx].reset_index(drop=True)

    if LABELING_MODE == "threshold":
        y = [_label_by_threshold(float(r), LABEL_SEUILS) for r in fut]
    else:
        ql = float(fut.quantile(QUANTILES["low"]))
        qh = float(fut.quantile(QUANTILES["high"]))
        y = [_label_by_quantile(float(r), ql, qh) for r in fut]

        tries, low, high = 0, QUANTILES["low"], QUANTILES["high"]
        cnt = Counter(y)
        while min(cnt.get("DOWN",0), cnt.get("RANGE",0), cnt.get("UP",0)) < MIN_PER_CLASS and tries < 5:
            low  = min(0.45, (low  + 0.50)/2)
            high = max(0.55, (high + 0.50)/2)
            ql = float(fut.quantile(low)); qh = float(fut.quantile(high))
            y = [_label_by_quantile(float(r), ql, qh) for r in fut]
            cnt = Counter(y); tries += 1

    dist = Counter(y)
    print("=== Distribution des labels (avant train) ===")
    for k in ["DOWN","RANGE","UP"]:
        print(f"{k}: {dist.get(k,0)}")

    y = pd.Series(y, name="label")
    return X, y, feature_cols
