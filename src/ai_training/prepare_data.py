import os
import pandas as pd
import ta  # librería de indicadores técnicos (pip install ta)

DATA_DIR = os.getenv("DATA_DIR", "./data")
SYMBOL = os.getenv("SYMBOL", "BTCUSDT")

def load_csv(symbol: str) -> pd.DataFrame:
    csv_path = os.path.join(DATA_DIR, f"ohlcv_{symbol}.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"No se encontró el CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    df.columns = ["timestamp", "open", "high", "low", "close", "volume"]
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    # Indicadores técnicos
    df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
    df["ema20"] = ta.trend.EMAIndicator(df["close"], window=20).ema_indicator()
    df["ema50"] = ta.trend.EMAIndicator(df["close"], window=50).ema_indicator()
    df["vol_ma20"] = df["volume"].rolling(window=20).mean()
    df["vol_ratio"] = df["volume"] / df["vol_ma20"]

    # Eliminamos filas con NaN por cálculos iniciales
    df = df.dropna()
    return df

def add_labels(df: pd.DataFrame, threshold: float = 0.003) -> pd.DataFrame:
    """
    Crea etiquetas de trading:
    - Buy si el precio sube más de threshold en la próxima vela
    - Sell si baja más de threshold
    - Hold en otro caso
    """
    df["future_return"] = df["close"].shift(-1) / df["close"] - 1
    conditions = [
        df["future_return"] > threshold,
        df["future_return"] < -threshold
    ]
    choices = ["buy", "sell"]
    df["label"] = pd.Series(pd.cut(df["future_return"],
                                   bins=[-float("inf"), -threshold, threshold, float("inf")],
                                   labels=["sell", "hold", "buy"]))
    df = df.dropna()
    return df

if __name__ == "__main__":
    df = load_csv(SYMBOL)
    df = add_features(df)
    df = add_labels(df)

    out_path = os.path.join(DATA_DIR, f"dataset_{SYMBOL}.csv")
    df.to_csv(out_path, index=False)
    print(f"✅ Dataset generado: {out_path}")
