# Version: V0.51 (Solution B + logging corrélation + seuils par horizon)
import os, time, traceback
from datetime import datetime, timezone
from collections import Counter

from config import (
    SYMBOL, TIMEFRAME, LOOKBACK_BARS_MINIMUM, MIN_BARS,
    REFRESH_INTERVAL, SMOOTH_K, ALERT_COOLDOWN_SEC, THRESHOLDS, CONF_PER_H,
    H1_TIMEFRAME, H1_LOOKBACK_BARS, HYSTERESIS_MARGIN,
    UP_IN_BEAR_MIN_CONF, DOWN_IN_BULL_MIN_CONF,
    LOG_DIR, STATE_PATH, PRINT_PROBA_PRECISION,
    MODEL_DIR, HORIZONS
)
from data.fetcher import get_ohlcv
from data.features import compute_features
from model.utils import load_model, load_schema, align_features
from model.predict import predict_trend
from alerts.notifier import console
from state.persistence import load_state, save_state, update_history
from analysis.logger import log_prediction

# -------- Utils --------
def now_str():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

def ensure_dirs():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs("state", exist_ok=True)

def fmt_top_probs(proba_dict, n=3):
    items = sorted(proba_dict.items(), key=lambda kv: kv[1], reverse=True)[:n]
    return ", ".join(f"{k}:{round(v*100, PRINT_PROBA_PRECISION)}%" for k, v in items)

def top2(proba_dict):
    items = sorted(proba_dict.items(), key=lambda kv: kv[1], reverse=True)
    if not items:
        return None, 0.0, None, 0.0
    (l1, p1) = items[0]
    (l2, p2) = items[1] if len(items) > 1 else (None, 0.0)
    return l1, p1, l2, p2

def apply_confidence_threshold(label, proba_dict, min_conf):
    if proba_dict.get(label, 0.0) >= min_conf:
        return label
    return "NEUTRE"

def majority_smoothing(history, current_label, k):
    arr = (history + [current_label])[-k:]
    if not arr:
        return current_label
    c = Counter(arr)
    return c.most_common(1)[0][0]

def cooldown_ok(last_ts, cooldown_sec):
    return (time.time() - last_ts) >= cooldown_sec

# -------- H1 regime & filters --------
def detect_h1_regime() -> str:
    df_h1 = get_ohlcv(SYMBOL, H1_TIMEFRAME, H1_LOOKBACK_BARS)
    if df_h1 is None or len(df_h1) < 210:
        return "NEUTRE"
    dfh = compute_features(df_h1)
    if dfh is None or len(dfh) == 0:
        return "NEUTRE"
    last = dfh.iloc[-1]
    close = float(last["close"]); sma200 = float(last["sma200"]); rsi = float(last["rsi14"])
    if close > sma200 and rsi >= 50: return "BULL"
    if close < sma200 and rsi <= 50: return "BEAR"
    return "NEUTRE"

def apply_hysteresis(prev_label: str, proba_dict: dict, margin: float) -> str:
    l1, p1, _, _ = top2(proba_dict)
    if prev_label is None or prev_label in ("", "UNKNOWN"):
        return l1
    p_prev = proba_dict.get(prev_label, 0.0)
    if l1 != prev_label and (p1 - p_prev) < margin:
        return prev_label
    return l1

def apply_h1_filter(label: str, proba_dict: dict, regime_h1: str) -> str:
    if regime_h1 == "BEAR" and label == "UP":
        if proba_dict.get("UP", 0.0) < UP_IN_BEAR_MIN_CONF:
            return "RANGE"
    if regime_h1 == "BULL" and label == "DOWN":
        if proba_dict.get("DOWN", 0.0) < DOWN_IN_BULL_MIN_CONF:
            return "RANGE"
    return label

# -------- Main --------
def main():
    ensure_dirs()
    state = load_state(STATE_PATH)

    models = {}; schemas = {}
    for H in HORIZONS:
        model_path = os.path.join(MODEL_DIR, f"xgb_H{H}.pkl")
        schema_path = os.path.join(MODEL_DIR, f"schema_H{H}.json")
        if not (os.path.exists(model_path) and os.path.exists(schema_path)):
            console(f"⚠ Modèle/Schéma manquant pour H={H}. Entraîne d'abord: python -m model.train_multi")
            return
        models[H] = load_model(model_path)
        schemas[H] = load_schema(schema_path)

    console(f"✅ Démarrage bot | TF={TIMEFRAME} | H={HORIZONS} | Modèles: {MODEL_DIR}")

    while True:
        try:
            df = get_ohlcv(SYMBOL, TIMEFRAME, LOOKBACK_BARS_MINIMUM)
            if df is None or len(df) < MIN_BARS:
                console("⚠ Données insuffisantes, on réessaie au prochain tick...")
                time.sleep(REFRESH_INTERVAL); continue

            df_feat = compute_features(df)
            if df_feat is None or len(df_feat) < MIN_BARS:
                console("⚠ Features insuffisantes, on réessaie au prochain tick...")
                time.sleep(REFRESH_INTERVAL); continue

            regime_h1 = detect_h1_regime()
            lines = []
            prev_short = state.get("last_by_h", {}).get(str(min(HORIZONS)), "UNKNOWN")

            for H in HORIZONS:
                model = models[H]; schema = schemas[H]
                feature_names = schema["feature_names"]; class_names = schema["classes_"]

                X = align_features(df_feat, {"feature_names": feature_names})
                x_now = X.iloc[[-1]]

                raw_label, proba_now = predict_trend(model, x_now, class_names)

                prev_label = state.get("last_by_h", {}).get(str(H), "UNKNOWN")
                label_hyst = apply_hysteresis(prev_label, proba_now, HYSTERESIS_MARGIN)
                label_h1 = apply_h1_filter(label_hyst, proba_now, regime_h1)

                hist = state.get("history_by_h", {}).get(str(H), [])
                smoothed_non_neutral = majority_smoothing(hist, label_h1, SMOOTH_K)

                min_conf = CONF_PER_H.get(H, THRESHOLDS["min_confidence"])
                display_label = apply_confidence_threshold(smoothed_non_neutral, proba_now, min_conf)

                curr_price = float(df_feat.iloc[-1]["close"])
                log_prediction(
                    symbol=SYMBOL, timeframe=TIMEFRAME, price=curr_price, horizon=H,
                    proba=proba_now, label_display=display_label, label_internal=label_h1,
                    regime_h1=regime_h1
                )

                update_history(state, label_h1, max_len=32, horizon=H)

                lines.append(f"  H={H:<3} → {display_label:<6} (top: {fmt_top_probs(proba_now)})")

            curr_short = state.get("last_by_h", {}).get(str(min(HORIZONS)), "UNKNOWN")
            if curr_short != prev_short:
                if cooldown_ok(state.get("dernier_timestamp_alerte", 0.0), ALERT_COOLDOWN_SEC):
                    console(f"⚠ Changement (H={min(HORIZONS)}) → {curr_short} | H1={regime_h1}")
                    state["dernier_timestamp_alerte"] = time.time()

            console(f"[{now_str()}] H1={regime_h1}\n" + "\n".join(lines))
            save_state(STATE_PATH, state)

        except Exception as e:
            console("❌ Erreur: " + str(e))
            traceback.print_exc()

        time.sleep(REFRESH_INTERVAL)

if __name__ == "__main__":
    main()