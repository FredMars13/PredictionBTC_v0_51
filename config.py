# Version: V0.51
import os

# --- Marché & données ---
SYMBOL = "BTC/USDT"
TIMEFRAME = "15m"
LOOKBACK_BARS_MINIMUM = 3000
MIN_BARS = 800

# --- Boucle temps réel ---
REFRESH_INTERVAL = 300  # 5 min
SMOOTH_K = 3
ALERT_COOLDOWN_SEC = 600

# --- Multi-horizon ---
HORIZONS = [4, 8, 16]   # ≈ 1h, 2h, 4h sur TF=15m

# --- Labellisation ---
LABELING_MODE = "quantile"  # "quantile" ou "threshold"
LABEL_SEUILS = {"up": 0.010, "down": -0.010, "range": 0.003}
QUANTILES = {"low": 0.30, "high": 0.70}
MIN_PER_CLASS = 50
RETRAIN_LOOKBACK_BARS = 10000

# --- Modèle ---
MODEL_DIR = "models"
THRESHOLDS = {"min_confidence": 0.55}

# Seuils d'affichage par horizon (optionnels). Si absent, on utilise THRESHOLDS["min_confidence"]
CONF_PER_H = {4: 0.53, 8: 0.57, 16: 0.62}

XGB_PARAMS = {
    "n_estimators": 400,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "objective": "multi:softprob",
    "eval_metric": "mlogloss",
    "random_state": 42,
}

# --- Filtre H1 + Hystérésis ---
H1_TIMEFRAME = "1h"
H1_LOOKBACK_BARS = 600
HYSTERESIS_MARGIN = 0.10
UP_IN_BEAR_MIN_CONF = 0.70
DOWN_IN_BULL_MIN_CONF = 0.70

# --- API ---
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "")

# --- Logs/état ---
LOG_DIR = "logs"
STATE_PATH = os.path.join("state", "bot_state.json")

# --- Divers ---
PRINT_PROBA_PRECISION = 2
