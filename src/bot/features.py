# src/bot/features.py
import pandas as pd
import numpy as np

def add_features(df):
    df = df.copy()
    df['close'] = df['close'].astype(float)
    df['logret'] = np.log(df['close'] / df['close'].shift(1))
    # Moving averages
    df['sma_5'] = df['close'].rolling(5).mean()
    df['sma_21'] = df['close'].rolling(21).mean()
    df['ema_9'] = df['close'].ewm(span=9, adjust=False).mean()
    df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
    # RSI
    delta = df['close'].diff()
    up = delta.clip(lower=0)
    down = -1*delta.clip(upper=0)
    ema_up = up.ewm(com=13, adjust=False).mean()
    ema_down = down.ewm(com=13, adjust=False).mean()
    rs = ema_up / (ema_down + 1e-9)
    df['rsi_14'] = 100 - (100/(1+rs))
    # MACD
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26
    df['macd_sig'] = df['macd'].ewm(span=9, adjust=False).mean()
    # Bollinger
    df['bb_mid'] = df['close'].rolling(20).mean()
    df['bb_std'] = df['close'].rolling(20).std()
    df['bb_width'] = (df['bb_std']*2) / df['bb_mid']
    # OBV
    obv = []
    obv_val = 0
    for i in range(1, len(df)):
        if df['close'].iat[i] > df['close'].iat[i-1]:
            obv_val += df['volume'].iat[i]
        elif df['close'].iat[i] < df['close'].iat[i-1]:
            obv_val -= df['volume'].iat[i]
        obv.append(obv_val)
    df['obv'] = pd.Series([np.nan] + obv)
    # VWAP
    df['typical'] = (df['high'] + df['low'] + df['close'])/3
    df['cum_vol_typ'] = (df['typical'] * df['volume']).cumsum()
    df['cum_vol'] = df['volume'].cumsum()
    df['vwap'] = df['cum_vol_typ'] / (df['cum_vol'] + 1e-9)
    df = df.dropna()
    return df
