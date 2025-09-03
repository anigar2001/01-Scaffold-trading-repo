import os
import pandas as pd

DATA_DIR = os.getenv("DATA_DIR", "/app/data")

def _csv_path(symbol: str, tf: str) -> str:
    return os.path.join(DATA_DIR, f"ohlcv_{symbol}_{tf}.csv")

def append_ohlcv_df(symbol: str, timeframe: str, df: pd.DataFrame):
    """
    df con columnas: timestamp, open, high, low, close, volume (timestamp en datetime)
    Añade filas nuevas (sin duplicar timestamps) y guarda ordenado.
    """
    path = _csv_path(symbol, timeframe)
    df = df.copy()
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df[['timestamp','open','high','low','close','volume']].dropna()

    if os.path.exists(path):
        old = pd.read_csv(path, parse_dates=['timestamp'])
        merged = pd.concat([old, df], ignore_index=True)
        merged = merged.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
    else:
        merged = df.sort_values('timestamp')

    merged.to_csv(path, index=False)

def resample_from_1m(symbol: str):
    """Genera 5m/15m/1h a partir de ohlcv_1m.csv"""
    path_1m = _csv_path(symbol, "1m")
    if not os.path.exists(path_1m):
        return

    df = pd.read_csv(path_1m, parse_dates=['timestamp']).set_index('timestamp').sort_index()
    # Resample OHLCV estándar
    def ohlc(tf):
        r = df['close'].resample(tf).ohlc()
        v = df['volume'].resample(tf).sum()
        out = r.join(v)
        out.columns = ['open','high','low','close','volume']
        out = out.dropna()
        out = out.reset_index().rename(columns={'index':'timestamp'})
        return out

    for tf, rule in [("5m","5min"), ("15m","15min"), ("1h","1h")]:
        out = ohlc(rule)
        out.to_csv(_csv_path(symbol, tf), index=False)
