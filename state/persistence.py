# Version: V0.51
import os, json

def load_state(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except:
                pass
    return {
        "dernière_tendance": "UNKNOWN",
        "dernier_timestamp_alerte": 0.0,
        "historique_labels": [],
        "last_by_h": {},
        "history_by_h": {}
    }

def save_state(path: str, state: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

def update_history(state: dict, label: str, max_len: int = 32, horizon: int = None):
    if horizon is None:
        hist = state.get("historique_labels", [])
        hist.append(label)
        if len(hist) > max_len:
            hist = hist[-max_len:]
        state["historique_labels"] = hist
    else:
        hb = state.get("history_by_h", {})
        key = str(horizon)
        arr = hb.get(key, [])
        arr.append(label)
        if len(arr) > max_len:
            arr = arr[-max_len:]
        hb[key] = arr
        state["history_by_h"] = hb
        lb = state.get("last_by_h", {})
        lb[key] = label
        state["last_by_h"] = lb
