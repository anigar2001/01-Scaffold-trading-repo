# src/connectors/ccxt_connector.py

import os
from typing import Dict, Any, Optional, List

import ccxt
from dotenv import load_dotenv

load_dotenv()


class CCXTConnector:
    """
    Conector CCXT para Binance (spot).
    - Acepta símbolos 'BTCUSDT' o 'BTC/USDT' y los normaliza a formato CCXT.
    - Soporta testnet (sandbox) con ajuste de hora y recvWindow.
    """

    QUOTE_CANDIDATES: List[str] = [
        "USDT", "BUSD", "USDC", "FDUSD",
        "EUR", "USD", "TRY", "BRL",
        "BTC", "ETH", "BNB", "SOL"
    ]

    def __init__(self):
        # Env
        api_key = os.getenv("BINANCE_API_KEY", "") or None
        api_secret = os.getenv("BINANCE_API_SECRET", "") or None
        use_testnet = (os.getenv("BINANCE_TESTNET", "true").lower() in ("1", "true", "yes"))

        # Construir exchange
        cfg: Dict[str, Any] = {
            "enableRateLimit": True,
            "options": {
                # muy importante para evitar -1021 (timestamp fuera de ventana)
                "adjustForTimeDifference": True,
                # ampliar ventana de recepción
                "recvWindow": 10_000,
                # tipo por defecto: spot
                "defaultType": "spot",
            },
        }
        if api_key and api_secret:
            cfg["apiKey"] = api_key
            cfg["secret"] = api_secret

        self.exchange = ccxt.binance(cfg)

        # Testnet/Sandbox
        if use_testnet:
            # CCXT soporta sandbox para binance spot
            self.exchange.set_sandbox_mode(True)

        # Cargar mercados (necesario para validar símbolos)
        self._ensure_markets_loaded()

    # -------------- Helpers --------------

    def _ensure_markets_loaded(self):
        """
        Carga markets si aún no están disponibles. Reintenta una vez si falla.
        """
        try:
            if not getattr(self.exchange, "markets", None):
                self.exchange.load_markets()
        except Exception:
            # reintento único
            self.exchange.load_markets()

    @classmethod
    def to_ccxt_symbol(cls, symbol: str) -> str:
        """
        Convierte 'BTCUSDT' -> 'BTC/USDT'. Si ya viene con barra, lo devuelve igual.
        Si no puede inferir la quote, devuelve tal cual y CCXT lanzará error claro.
        """
        if not symbol:
            return symbol
        s = symbol.strip().upper()
        if "/" in s:
            return s

        # intenta dividir por quotes conocidas: el sufijo es la quote
        for q in cls.QUOTE_CANDIDATES:
            if s.endswith(q) and len(s) > len(q):
                base = s[:-len(q)]
                # casos como BTCBUSD, ETHUSDT, etc.
                return f"{base}/{q}"

        # fallback: devolver sin barra (puede fallar en CCXT, pero con mensaje claro)
        return s

    # -------------- API Pública --------------

    def get_balance(self) -> Dict[str, float]:
        """
        Devuelve un dict con balances totales por moneda, ej: {'USDT': 1000.0, 'BTC': 0.5}
        """
        try:
            self._ensure_markets_loaded()
            bal = self.exchange.fetch_balance()
            # 'total' contiene el saldo total por moneda
            return bal.get("total", {})
        except ccxt.BaseError as e:
            raise RuntimeError(f"Error al obtener balance: {e}")

    def fetch_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        Ticker unificado de CCXT. Acepta BTCUSDT o BTC/USDT.
        """
        try:
            self._ensure_markets_loaded()
            sym = self.to_ccxt_symbol(symbol)
            # Validación de símbolo (si es posible)
            if getattr(self.exchange, "markets", None) and sym not in self.exchange.markets:
                # Si vino sin barra y no se pudo inferir bien, al menos lanzamos un error comprensible
                raise ValueError(f"Símbolo no disponible en Binance: '{symbol}' (normalizado: '{sym}')")

            t = self.exchange.fetch_ticker(sym)
            # t típicamente incluye: symbol, bid, ask, last, baseVolume, quoteVolume, timestamp, datetime, etc.
            return {
                "symbol": t.get("symbol", sym),
                "bid": t.get("bid"),
                "ask": t.get("ask"),
                "last": t.get("last"),
                "timestamp": t.get("timestamp"),
                "datetime": t.get("datetime"),
                "baseVolume": t.get("baseVolume"),
                "quoteVolume": t.get("quoteVolume"),
                "info": t.get("info", {}),
            }
        except ccxt.BaseError as e:
            raise RuntimeError(f"Error al obtener ticker {symbol}: {e}")
        except Exception as e:
            raise RuntimeError(f"Error al obtener ticker {symbol}: {e}")

    def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        Alias legado para compatibilidad (algunos módulos llaman get_ticker).
        """
        return self.fetch_ticker(symbol)

    def fetch_ohlcv(self, symbol: str, timeframe: str = "1m", limit: int = 200) -> list:
        self._ensure_markets_loaded()
        sym = self.to_ccxt_symbol(symbol)
        if getattr(self.exchange, "markets", None) and sym not in self.exchange.markets:
            raise ValueError(f"Símbolo no disponible en Binance: '{symbol}' (normalizado: '{sym}')")
        data = self.exchange.fetch_ohlcv(sym, timeframe=timeframe, limit=limit)
        # 'data' es lista; aquí sí se puede usar len()
        if data is None or len(data) == 0:
            raise RuntimeError("fetch_ohlcv devolvió vacío.")
        return data


    # -------------- Órdenes --------------

    def create_order(
        self,
        symbol: str,
        side: str,
        type: str,
        amount: float,
        price: Optional[float] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Crea una orden market o limit.
        side: 'buy' | 'sell'
        type: 'market' | 'limit'
        """
        try:
            self._ensure_markets_loaded()
            sym = self.to_ccxt_symbol(symbol)
            if getattr(self.exchange, "markets", None) and sym not in self.exchange.markets:
                raise ValueError(f"Símbolo no disponible en Binance: '{symbol}' (normalizado: '{sym}')")

            side = side.lower().strip()
            type = type.lower().strip()
            params = params or {}

            if type == "market":
                order = self.exchange.create_market_order(sym, side, amount, params=params)
            elif type == "limit":
                if price is None:
                    raise ValueError("Para órdenes 'limit' se requiere 'price'.")
                order = self.exchange.create_limit_order(sym, side, amount, price, params=params)
            else:
                raise ValueError("Tipo de orden no soportado. Use 'market' o 'limit'.")

            return order
        except ccxt.BaseError as e:
            raise RuntimeError(f"Error al crear orden {symbol}: {e}")
        except Exception as e:
            raise RuntimeError(f"Error al crear orden {symbol}: {e}")

