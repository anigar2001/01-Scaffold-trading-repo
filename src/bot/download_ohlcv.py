import ccxt
import pandas as pd
from datetime import datetime
import os

# Configuración
SYMBOL = 'BTCUSDT'
TIMEFRAME = '1m'  # vela de 1 minuto
LIMIT = 1000      # cantidad de velas por request
DATA_DIR = '/app/data'
os.makedirs(DATA_DIR, exist_ok=True)
CSV_FILE = os.path.join(DATA_DIR, f'{SYMBOL}_ohlcv.csv')

# Inicializar exchange Testnet
exchange = ccxt.binance({
    'apiKey': os.getenv('BINANCE_API_KEY'),
    'secret': os.getenv('BINANCE_API_SECRET'),
    'enableRateLimit': True,
})
exchange.set_sandbox_mode(True)

# Descargar OHLCV
ohlcv = exchange.fetch_ohlcv(SYMBOL, timeframe=TIMEFRAME, limit=LIMIT)

# Convertir a DataFrame
df = pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

# Guardar CSV
df.to_csv(CSV_FILE, index=False)
print(f"Datos OHLCV guardados en {CSV_FILE}")
