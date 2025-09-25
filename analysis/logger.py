# Version: V0.51
import os, csv
from datetime import datetime, timezone

LOG_PATH = os.path.join("logs", "preds.csv")

HEADER = [
    "ts_utc","symbol","timeframe","price","horizon",
    "p_up","p_range","p_down",
    "top_label","top_prob",
    "label_display","label_internal","regime_h1"
]

def _ensure_dir(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

def _now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

def log_prediction(symbol: str, timeframe: str, price: float,
                   horizon: int, proba: dict, label_display: str,
                   label_internal: str, regime_h1: str,
                   out_path: str = LOG_PATH):
    _ensure_dir(out_path)
    top_label = max(proba, key=proba.get)
    top_prob = float(proba[top_label])
    row = [
        _now_iso(), symbol, timeframe, float(price), int(horizon),
        float(proba.get("UP", 0.0)),
        float(proba.get("RANGE", 0.0)),
        float(proba.get("DOWN", 0.0)),
        top_label, top_prob,
        label_display, label_internal, regime_h1
    ]
    new_file = not os.path.exists(out_path)
    with open(out_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(HEADER)
        w.writerow(row)
