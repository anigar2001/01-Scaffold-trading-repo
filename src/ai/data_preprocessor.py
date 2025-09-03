import os
import pandas as pd
import ta  # librería técnica
import joblib

DATA_DIR = "/app/data"
OUTPUT_FILE = os.path.join(DATA_DIR, "combined_features.pkl")

# ======================
# Funciones auxiliares
# ======================
def load_csv(symbol: str, timeframe: str) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, f"ohlcv_{symbol}_{timeframe}.csv")
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df.set_index("timestamp", inplace=True)
    df = df.sort_index()
    return df

def add_indicators(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Añade indicadores técnicos y los renombra con un prefijo según timeframe"""
    df[f"{prefix}_ema_10"] = ta.trend.EMAIndicator(df["close"], window=10).ema_indicator()
    df[f"{prefix}_ema_50"] = ta.trend.EMAIndicator(df["close"], window=50).ema_indicator()
    df[f"{prefix}_rsi_14"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
    macd = ta.trend.MACD(df["close"])
    df[f"{prefix}_macd"] = macd.macd()
    df[f"{prefix}_macd_signal"] = macd.macd_signal()
    df[f"{prefix}_macd_diff"] = macd.macd_diff()
    df[f"{prefix}_vol_ma_20"] = df["volume"].rolling(window=20).mean()
    return df

# ======================
# Proceso principal
# ======================
def build_combined_dataset(symbol: str = "BTCUSDT"):
    # Cargar cada timeframe
    df_1m = load_csv(symbol, "1m")
    df_5m = load_csv(symbol, "5m")
    df_15m = load_csv(symbol, "15m")
    df_1h = load_csv(symbol, "1h")

    # Añadir indicadores
    df_1m = add_indicators(df_1m, "t1m")
    df_5m = add_indicators(df_5m, "t5m")
    df_15m = add_indicators(df_15m, "t15m")
    df_1h = add_indicators(df_1h, "t1h")

    # Resamplear los de mayor timeframe a 1m y rellenar hacia delante
    df_5m = df_5m.resample("1min").ffill()
    df_15m = df_15m.resample("1min").ffill()
    df_1h = df_1h.resample("1min").ffill()

    # Unir todo en índice 1m
    combined = df_1m.join(df_5m, rsuffix="_5m", how="inner")
    combined = combined.join(df_15m, rsuffix="_15m", how="inner")
    combined = combined.join(df_1h, rsuffix="_1h", how="inner")

    # Eliminar nulos iniciales
    combined = combined.dropna()

    # Guardar en pkl
    joblib.dump(combined, OUTPUT_FILE)
    print(f"✅ Dataset combinado guardado en {OUTPUT_FILE} con shape {combined.shape}")

if __name__ == "__main__":
    build_combined_dataset()
