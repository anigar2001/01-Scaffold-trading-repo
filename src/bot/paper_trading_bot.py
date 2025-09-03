# src/bot/paper_trading_bot.py

import os
import time
import pandas as pd
from dotenv import load_dotenv

from src.connectors.ccxt_connector import CCXTConnector
from src.connectors.mock_exchange import MockExchange
from src.order_manager import OrderManager
from src.utils.logger import log_trade
from src.bot.ai_signal import AISignal
from src.utils.ohlcv_store import append_ohlcv_df, resample_from_1m

load_dotenv()

print("Bot arrancando...")

# ==========================
# Config
# ==========================
SYMBOL = os.getenv("BOT_SYMBOL", "BTCUSDT")       # Binance testnet: sin barra
TIMEFRAME = os.getenv("BOT_TIMEFRAME", "1m")
TRADE_AMOUNT = float(os.getenv("BOT_TRADE_AMOUNT", "0.001"))
SLEEP_INTERVAL = int(os.getenv("BOT_SLEEP_SEC", "60"))
HISTORY_LIMIT = int(os.getenv("BOT_HISTORY_LIMIT", "100"))
CONNECTOR_TYPE = os.getenv("CONNECTOR_TYPE", "MOCK").upper()

# ==========================
# Conector
# ==========================
if CONNECTOR_TYPE == "CCXT":
    connector = CCXTConnector()
else:
    connector = MockExchange()

order_manager = OrderManager(connector)
# PASAR el conector al AISignal (IMPRESCINDIBLE)
ai = AISignal(connector=connector)

# ==========================
# Helpers
# ==========================
def fetch_ohlcv_df(symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
    symbol = symbol.replace("/", "").upper()  # BTC/USDT -> BTCUSDT
    ohlcv = connector.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    if not ohlcv or len(ohlcv) == 0:
        raise RuntimeError(f"OHLCV vacío para {symbol} {timeframe}")
    df = pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])
    if pd.api.types.is_integer_dtype(df["timestamp"]) or pd.api.types.is_float_dtype(df["timestamp"]):
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df[['open','high','low','close','volume']] = df[['open','high','low','close','volume']].astype(float)
    # guarda 1m
    append_ohlcv_df(SYMBOL, "1m", df[['timestamp','open','high','low','close','volume']])
    # genera 5m/15m/1h a partir de 1m
    resample_from_1m(SYMBOL)    
    return df

# ==========================
# Estado inicial
# ==========================
position = "flat"  # 'flat', 'long', 'short'

# Test inicial de fetch
try:
    _df_test = fetch_ohlcv_df(SYMBOL, TIMEFRAME, 10)
    print(f"Fetch OHLCV OK: {SYMBOL} {TIMEFRAME} -> {len(_df_test)} velas")
except Exception as e:
    print("ERROR de inicialización al obtener OHLCV:", e)

print("Paper Trading Bot con IA y hold inteligente iniciado...")

# ==========================
# Loop principal
# ==========================
while True:
    try:
        # (opcional) traer df por si quieres loguear último precio
        df = fetch_ohlcv_df(SYMBOL, TIMEFRAME, HISTORY_LIMIT)
        
        try:
            t = connector.fetch_ticker(SYMBOL)
            live_last = float(t.get('last') or t.get('close') or t.get('bid') or last_price)
            last_price = live_last  # usa el más fresco para logging/decisión
        except Exception:
            pass
        
        last_ts = df['timestamp'].iloc[-1]
        last_price = float(df['close'].iloc[-1])
        #print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {SYMBOL} ts={last_ts} last={last_price}")


        # *** Señal IA multitimeframe ***
        # IMPORTANTE: ahora la IA espera SYMBOL (y usa el conector internamente)
        signal_dict = ai.predict_signal(SYMBOL)
        signal_type = str(signal_dict.get("signal", "hold")).lower()
        confidence = float(signal_dict.get("confidence", 0.0))
        # conserva el precio del df como autoridad; usa el de IA solo si no hay
        ia_last = signal_dict.get("last_price", None)
        if ia_last is not None:
            # opcional: verifica que el timestamp 1m coincide
            last_price = float(ia_last)

        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {SYMBOL} Last: {last_price}, IA Signal: {signal_type} ({confidence}%)")

        # Reglas de posición
        action = "hold"
        if signal_type == "buy":
            if position == "flat":
                action = "buy"; position = "long"
            elif position == "short":
                action = "sell"; position = "flat"  # cierra short
        elif signal_type == "sell":
            if position == "flat":
                action = "sell"; position = "short"
            elif position == "long":
                action = "sell"; position = "flat"  # cierra long

        print(f"Acción: {action} | Posición: {position}")

        # Ejecutar orden si no es hold
        if action != "hold":
            order = order_manager.create_order(
                symbol=SYMBOL,
                side=action,
                type="market",
                amount=TRADE_AMOUNT
            )
            print("Orden ejecutada:", order)

            # Registrar operación
            log_trade(
                symbol=SYMBOL,
                side=action,
                price=last_price,
                amount=TRADE_AMOUNT,
                order_id=order.get("id", "")
            )

    except Exception as e:
        print("Error en bot:", e)

    time.sleep(SLEEP_INTERVAL)


