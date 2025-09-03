# src/utils/logger.py

import os
import csv
from datetime import datetime

# Directorio dentro del contenedor
DATA_DIR = "/app/data"
os.makedirs(DATA_DIR, exist_ok=True)

LOG_FILE = os.path.join(DATA_DIR, "trades.csv")

def log_trade(symbol, side, price, amount, order_id):
    """
    Guarda cada operaciÃ³n ejecutada en un CSV compartido con el host.
    """
    file_exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, mode="a", newline="") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["timestamp", "symbol", "side", "price", "amount", "order_id"])
        writer.writerow([
            datetime.utcnow().isoformat(),
            symbol,
            side,
            price,
            amount,
            order_id
        ])


def log_trade_ex(symbol, side, price, amount, order_id,
                 signal=None, confidence=None, reason: str | None = None):
    """
    Variante extendida que añade columnas: signal, confidence, reason.
    Si el CSV existente es antiguo, escribe solo las columnas clásicas.
    """
    headers_full = [
        "timestamp", "symbol", "side", "price", "amount", "order_id",
        "signal", "confidence", "reason"
    ]
    headers_legacy = ["timestamp", "symbol", "side", "price", "amount", "order_id"]

    file_exists = os.path.isfile(LOG_FILE)
    # ¿CSV legacy?
    write_full = True
    if file_exists:
        try:
            with open(LOG_FILE, mode="r", newline="") as rf:
                first = rf.readline()
                if first and (",reason" not in first):
                    write_full = False
        except Exception:
            write_full = True

    with open(LOG_FILE, mode="a", newline="") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(headers_full)
        row_full = [
            datetime.utcnow().isoformat(), symbol, side, price, amount, order_id,
            signal, confidence, reason
        ]
        if write_full:
            writer.writerow(row_full)
        else:
            writer.writerow(row_full[:len(headers_legacy)])

