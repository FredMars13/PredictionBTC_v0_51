# Version: V0.51
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator, MACD

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["open","high","low","close","volume"]:
        df[col] = df[col].astype(float)

    df["sma50"] = SMAIndicator(df["close"], window=50).sma_indicator()
    df["sma100"] = SMAIndicator(df["close"], window=100).sma_indicator()
    df["sma200"] = SMAIndicator(df["close"], window=200).sma_indicator()

    df["rsi14"] = RSIIndicator(df["close"], window=14).rsi()

    macd = MACD(df["close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()

    df["ret1"] = df["close"].pct_change()
    df["ret5"] = df["close"].pct_change(5)
    df["ret10"] = df["close"].pct_change(10)
    df["hl_spread"] = (df["high"] - df["low"]) / df["close"]
    df["oc_spread"] = (df["close"] - df["open"]) / df["open"]

    for k in [1,2,3,4,5]:
        df[f"rsi14_lag{k}"] = df["rsi14"].shift(k)
        df[f"ret1_lag{k}"] = df["ret1"].shift(k)

    df = df.dropna().reset_index(drop=True)
    return df
