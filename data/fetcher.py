# Version: V0.51
import ccxt
import pandas as pd

def get_ohlcv(symbol: str, timeframe: str, lookback_bars: int = 1500):
    try:
        ex = ccxt.binance()
        data = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=lookback_bars)
        if not data:
            return None
        df = pd.DataFrame(
            data, columns=["time","open","high","low","close","volume"]
        )
        df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
        return df
    except Exception:
        return None
