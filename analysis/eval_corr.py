# Version: V0.51
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ccxt

from config import SYMBOL, TIMEFRAME

LOG_PATH = os.path.join("logs", "preds.csv")
OUT_DIR = os.path.join("logs", "analysis")

def load_logs():
    if not os.path.exists(LOG_PATH):
        raise FileNotFoundError(f"Fichier introuvable: {LOG_PATH}")
    df = pd.read_csv(LOG_PATH, parse_dates=["ts_utc"])
    return df.sort_values("ts_utc").reset_index(drop=True)

def fetch_ohlcv(symbol: str, timeframe: str, since_ms=None, limit=10000):
    ex = ccxt.binance()
    data = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=since_ms, limit=limit)
    df = pd.DataFrame(data, columns=["ts_ms","open","high","low","close","volume"])
    df["ts"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
    return df

def round_to_tf(ts: pd.Timestamp, timeframe: str) -> pd.Timestamp:
    ts = ts.tz_convert("UTC")
    if timeframe.endswith("m"):
        m = int(timeframe[:-1]); mm = (ts.minute // m) * m
        return ts.replace(minute=mm, second=0, microsecond=0)
    if timeframe.endswith("h"):
        h = int(timeframe[:-1]); hh = (ts.hour // h) * h
        return ts.replace(hour=hh, minute=0, second=0, microsecond=0)
    return ts.replace(second=0, microsecond=0)

def build_future_returns(df_logs: pd.DataFrame, df_ohlcv: pd.DataFrame) -> pd.DataFrame:
    o = df_ohlcv.set_index("ts")[["close"]].sort_index()
    rows = []
    for _, r in df_logs.iterrows():
        t0 = round_to_tf(r["ts_utc"].tz_localize("UTC"), r["timeframe"])
        H = int(r["horizon"])
        if r["timeframe"].endswith("m"):
            m = int(r["timeframe"][:-1]); delta = pd.Timedelta(minutes=m*H)
        elif r["timeframe"].endswith("h"):
            h = int(r["timeframe"][:-1]); delta = pd.Timedelta(hours=h*H)
        else:
            delta = pd.Timedelta(minutes=15*H)
        t1 = t0 + delta
        try:
            p0 = float(o.loc[t0, "close"]); p1 = float(o.loc[t1, "close"])
            fut = (p1/p0) - 1.0
        except KeyError:
            p0 = np.nan; p1 = np.nan; fut = np.nan
        rows.append({**r.to_dict(), "t_candle": t0, "t_future": t1,
                     "price_candle": p0, "price_future": p1, "future_ret": fut})
    return pd.DataFrame(rows)

def realized_label_from_return(ret: float, eps: float = 0.0) -> str:
    if pd.isna(ret): return "NA"
    if ret > eps: return "UP"
    if ret < -eps: return "DOWN"
    return "RANGE"

def evaluate(df: pd.DataFrame):
    df = df.dropna(subset=["future_ret"]).copy()
    df["score_ud"] = df["p_up"] - df["p_down"]
    print(f"Corr(score_ud, future_ret) = {df['score_ud'].corr(df['future_ret']):.3f}")

    rows = []
    from sklearn.metrics import f1_score
    for H in sorted(df["horizon"].unique()):
        d = df[df["horizon"] == H].copy()
        d["y_true"] = [realized_label_from_return(x, 0.0) for x in d["future_ret"]]
        d = d[d["y_true"].isin(["UP","DOWN","RANGE"])]
        if d.empty:
            rows.append({"H":H,"n":0,"acc_top":None,"f1_top":None,"acc_display":None,"f1_display":None})
            continue
        def argmax_label(row):
            probs = {"UP": row["p_up"], "RANGE": row["p_range"], "DOWN": row["p_down"]}
            return max(probs, key=probs.get)
        d["y_pred_top"] = d.apply(argmax_label, axis=1)
        acc_top = (d["y_pred_top"] == d["y_true"]).mean()
        f1_top = f1_score(d["y_true"], d["y_pred_top"], average="macro", labels=["DOWN","RANGE","UP"])

        acc_disp = (d["label_display"] == d["y_true"]).mean()
        f1_disp = f1_score(d["y_true"], d["label_display"], average="macro", labels=["DOWN","RANGE","UP"])
        rows.append({"H":H,"n":len(d),"acc_top":acc_top,"f1_top":f1_top,"acc_display":acc_disp,"f1_display":f1_disp})
    print(pd.DataFrame(rows))

    os.makedirs(OUT_DIR, exist_ok=True)
    for H in sorted(df["horizon"].unique()):
        d = df[df["horizon"] == H].copy()
        d["y_true_up"] = (d["future_ret"] > 0).astype(int)
        bins = np.linspace(0,1,11)
        d["bin"] = pd.cut(d["p_up"], bins=bins, include_lowest=True)
        grp = d.groupby("bin", observed=False).agg(p_up_mean=("p_up","mean"), up_rate=("y_true_up","mean"), count=("y_true_up","size")).reset_index()
        grp = grp.dropna()
        plt.figure()
        plt.plot(grp["p_up_mean"], grp["up_rate"], marker="o")
        plt.plot([0,1],[0,1],"--")
        plt.xlabel("Proba prédite UP"); plt.ylabel("Fréquence réalisée UP")
        plt.title(f"Calibration UP – H={H}")
        plt.grid(True)
        plt.savefig(os.path.join(OUT_DIR, f"calibration_up_H{H}.png"), bbox_inches="tight")
        plt.close()

    for H in sorted(df["horizon"].unique()):
        d = df[df["horizon"] == H].copy()
        d = d.dropna(subset=["future_ret"])
        if len(d) < 5: continue
        s = d["future_ret"].rolling(50, min_periods=5).std()
        scale = s.median() if s.median() and not np.isnan(s.median()) else 1.0
        plt.figure()
        plt.plot(d["t_candle"], d["score_ud"], label="score_ud (p_up - p_down)")
        plt.plot(d["t_candle"], d["future_ret"]/scale, label=f"future_ret / {scale:.4f}")
        plt.axhline(0, linestyle="--", linewidth=1)
        plt.legend(); plt.grid(True)
        plt.title(f"Score vs Retour futur – H={H}")
        plt.xlabel("Temps (UTC)")
        plt.savefig(os.path.join(OUT_DIR, f"timeseries_corr_H{H}.png"), bbox_inches="tight")
        plt.close()

def main():
    df_logs = load_logs()
    tmin = df_logs["ts_utc"].min().tz_localize("UTC")
    since_ms = int((tmin - pd.Timedelta(days=5)).timestamp() * 1000)
    df_ohlcv = fetch_ohlcv(SYMBOL, TIMEFRAME, since_ms=since_ms, limit=10000)
    df = build_future_returns(df_logs, df_ohlcv)
    evaluate(df)
    os.makedirs(OUT_DIR, exist_ok=True)
    out_csv = os.path.join(OUT_DIR, "preds_with_future.csv")
    df.to_csv(out_csv, index=False)
    print(f"✅ Résultats écrits dans: {OUT_DIR}")
    print(f"🗂  Merge complet: {out_csv}")

if __name__ == "__main__":
    main()
