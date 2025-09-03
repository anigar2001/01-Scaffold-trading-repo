# src/bot/download_ohlcv_full.py
import ccxt, time, pandas as pd, os
from datetime import datetime, timedelta

exchange = ccxt.binance({
    'apiKey': os.getenv('BINANCE_API_KEY'),
    'secret': os.getenv('BINANCE_API_SECRET'),
    'enableRateLimit': True,
})
exchange.set_sandbox_mode(True)

DATA_DIR = "/app/data"
os.makedirs(DATA_DIR, exist_ok=True)

def timeframe_to_ms(tf):
    # mapping common timeframes
    mapping = {'1m': 60_000, '5m': 5*60_000, '15m':15*60_000,'1h':3600_000}
    return mapping[tf]

def fetch_all_ohlcv(symbol='BTCUSDT', timeframe='1m', since=None, limit=1000, max_candles=200000):
    all_ = []
    now = exchange.milliseconds()
    tf_ms = timeframe_to_ms(timeframe)
    if since is None:
        # fetch last max_candles by setting since to now - max_candles*tf_ms
        since = now - max_candles * tf_ms
    while True:
        try:
            data = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
        except Exception as e:
            print("Error fetch:", e)
            time.sleep(1)
            continue
        if not data:
            break
        all_.extend(data)
        last = data[-1][0]
        since = last + 1
        # rate limit safe sleep
        time.sleep(exchange.rateLimit/1000)
        if len(data) < limit:
            break
    df = pd.DataFrame(all_, columns=['timestamp','open','high','low','close','volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

if __name__ == "__main__":
    sym = 'BTCUSDT'
    for tf in ['1m','5m','15m','1h']:
        print("Downloading", sym, tf)
        df = fetch_all_ohlcv(symbol=sym, timeframe=tf, max_candles=200000)
        fname = os.path.join(DATA_DIR, f"ohlcv_{sym}_{tf}.csv")
        df.to_csv(fname, index=False)
        print("Saved", fname, "rows:", len(df))
