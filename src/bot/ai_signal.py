# src/bot/ai_signal.py

from __future__ import annotations

import os
import json
from typing import Tuple, Optional, Dict

import numpy as np
import pandas as pd
import joblib
import ta  # asegúrate de tener 'ta' en requirements (p.ej. ta==0.11.0)

# Rutas por defecto (puedes sobreescribir con variables de entorno)
DATA_DIR = os.getenv("DATA_DIR", "/app/data")
MODEL_FILE = os.getenv("AI_MODEL_FILE", os.path.join(DATA_DIR, "ai_model.pkl"))
FEATURES_FILE = os.getenv("AI_FEATURES_FILE", os.path.join(DATA_DIR, "ai_feature_names.json"))


class AISignal:
    """
    Genera una señal IA combinando timeframes 1m, 5m, 15m, 1h en tiempo real.

    - Se inyecta un 'connector' (CCXTConnector o Mock) con método:
         fetch_ohlcv(symbol, timeframe="1m", limit=N) -> list[list]
    - Carga el modelo entrenado (sklearn compatible con predict_proba)
      y el listado de features exactas que se usaron en el entrenamiento
      (ai_feature_names.json).
    """

    def __init__(self, connector=None):
        self.connector = connector

        if not os.path.exists(MODEL_FILE):
            raise FileNotFoundError(f"Modelo no encontrado: {MODEL_FILE}. Entrena primero.")
        if not os.path.exists(FEATURES_FILE):
            raise FileNotFoundError(
                f"Lista de features no encontrada: {FEATURES_FILE}. "
                f"Reentrena guardando las columnas de entrenamiento."
            )

        self.model = joblib.load(MODEL_FILE)
        with open(FEATURES_FILE, "r") as f:
            self.feature_cols = json.load(f)

        if not isinstance(self.feature_cols, list) or len(self.feature_cols) == 0:
            raise ValueError("ai_feature_names.json no contiene una lista válida de columnas de features.")

    # ─────────────────────────────
    # Helpers de features/indicadores
    # ─────────────────────────────
    def _add_indicators(self, df: pd.DataFrame, prefix: str) -> pd.DataFrame:
        """
        Misma lógica que en tu preprocesador: EMA(10/50), RSI(14), MACD(12/26/9), vol_ma(20).
        df debe venir con columnas: open, high, low, close, volume y index de datetime.
        """
        df = df.copy()

        # EMAs
        df[f"{prefix}_ema_10"] = ta.trend.EMAIndicator(df["close"], window=10).ema_indicator()
        df[f"{prefix}_ema_50"] = ta.trend.EMAIndicator(df["close"], window=50).ema_indicator()

        # RSI
        df[f"{prefix}_rsi_14"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()

        # MACD 12/26/9
        macd_obj = ta.trend.MACD(df["close"])
        df[f"{prefix}_macd"] = macd_obj.macd()
        df[f"{prefix}_macd_signal"] = macd_obj.macd_signal()
        df[f"{prefix}_macd_diff"] = macd_obj.macd_diff()

        # Volumen media 20
        df[f"{prefix}_vol_ma_20"] = df["volume"].rolling(window=20, min_periods=1).mean()

        return df

    def _fetch_ohlcv_df(self, symbol: str, timeframe: str, limit: int = 500) -> pd.DataFrame:
        """
        Pide OHLCV al conector y devuelve DF con:
          - index = timestamp tz-aware (UTC)
          - columnas: open, high, low, close, volume (floats)
        """
        if self.connector is None:
            raise RuntimeError("AISignal necesita un 'connector' inyectado con fetch_ohlcv().")

        ohlcv = self.connector.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        if ohlcv is None or len(ohlcv) < 50:
            raise RuntimeError(f"Datos insuficientes {symbol} {timeframe} (len={0 if ohlcv is None else len(ohlcv)})")

        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])

        # CCXT suele entregar ts en ms epoch → convertir a UTC tz-aware
        if pd.api.types.is_integer_dtype(df["timestamp"]) or pd.api.types.is_float_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        else:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

        # Tipado numérico estricto
        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        df = df.dropna(subset=["timestamp", "close"]).set_index("timestamp").sort_index()

        if df.empty:
            raise RuntimeError(f"OHLCV vacío/inválido para {symbol} {timeframe}")

        return df

    def _build_live_combined(self, symbol: str) -> Tuple[pd.DataFrame, pd.Timestamp, float]:
        """
        Construye el dataset combinado sobre índice 1m:
          - Descarga 1m, 5m, 15m, 1h en vivo
          - Calcula indicadores por TF
          - Re-muestrea 5m/15m/1h a 1m (ffill)
          - Une todo y elimina NaNs
        Devuelve: (combined_df, last_ts_1m [UTC], last_price_1m)
        """
        # 1) Descargas por timeframe
        d1m = self._fetch_ohlcv_df(symbol, "1m", limit=1000)
        d5m = self._fetch_ohlcv_df(symbol, "5m", limit=1000)
        d15m = self._fetch_ohlcv_df(symbol, "15m", limit=1000)
        d1h = self._fetch_ohlcv_df(symbol, "1h", limit=1000)

        last_ts_1m = d1m.index[-1]  # tz-aware UTC
        last_price_1m = float(d1m["close"].iloc[-1])

        # 2) Indicadores por TF
        d1m = self._add_indicators(d1m, "t1m")
        d5m = self._add_indicators(d5m, "t5m")
        d15m = self._add_indicators(d15m, "t15m")
        d1h = self._add_indicators(d1h, "t1h")

        # 3) Re-sample a 1m para alinear
        d5m = d5m.resample("1min").ffill()
        d15m = d15m.resample("1min").ffill()
        d1h = d1h.resample("1min").ffill()

        # 4) Joins
        combined = d1m.join(d5m, how="inner", rsuffix="_5m")
        combined = combined.join(d15m, how="inner", rsuffix="_15m")
        combined = combined.join(d1h, how="inner", rsuffix="_1h")

        # Limpieza final
        combined = combined.replace([np.inf, -np.inf], np.nan).dropna(how="any")

        return combined, last_ts_1m, last_price_1m

    # ─────────────────────────────
    # Predicción pública
    # ─────────────────────────────
    def predict_signal(self, symbol: str) -> Dict[str, object]:
        """
        Genera la señal IA en vivo para 'symbol'.
        Devuelve dict:
          {
            "signal": "buy"|"sell"|"hold",
            "confidence": float,     # margen % (best - second)
            "raw_conf": float,       # prob. % de la clase ganadora
            "last_price": float,
            "last_ts": ISO8601 UTC,
            "probs": {"-1": p1, "0": p2, "1": p3}
          }
        """
        try:
            combined, last_ts_1m, last_price_1m = self._build_live_combined(symbol)
        except Exception as e:
            # Fallback suave si no podemos construir el combinado (sin tumbar la API)
            return {
                "signal": "hold",
                "confidence": 0.0,
                "raw_conf": 0.0,
                "last_price": None,
                "last_ts": None,
                "error": f"build_live_failed: {e}",
            }

        if combined is None or combined.empty:
            return {
                "signal": "hold",
                "confidence": 0.0,
                "raw_conf": 0.0,
                "last_price": last_price_1m,
                "last_ts": last_ts_1m.isoformat(),
            }

        # Asegurar EXACTAMENTE las columnas de entrenamiento
        X = combined.copy()
        for c in self.feature_cols:
            if c not in X.columns:
                X[c] = np.nan
        X = X[self.feature_cols].astype(float).tail(1)

        if X.isna().any(axis=None):
            return {
                "signal": "hold",
                "confidence": 0.0,
                "raw_conf": 0.0,
                "last_price": last_price_1m,
                "last_ts": last_ts_1m.isoformat(),
                "error": "NaNs en features",
            }

        # Predicción
        try:
            if not hasattr(self.model, "predict_proba"):
                # Modelo sin probas → usar predict y devolver conf=0
                pred = self.model.predict(X)[0]
                signal_map = {-1: "sell", 0: "hold", 1: "buy"}
                label = signal_map.get(int(pred), "hold")
                return {
                    "signal": label,
                    "confidence": 0.0,
                    "raw_conf": 0.0,
                    "last_price": last_price_1m,
                    "last_ts": last_ts_1m.isoformat(),
                    "probs": {},
                }

            proba = self.model.predict_proba(X)[0]  # shape (n_classes,)
            # Orden por prob. descendente
            order = np.argsort(proba)[::-1]
            best_idx = int(order[0])
            second_idx = int(order[1]) if len(order) > 1 else best_idx

            # Clases del modelo (ej. [-1, 0, 1])
            classes = getattr(self.model, "classes_", np.array([0, 1]))
            best_label = int(classes[best_idx])

            signal_map = {-1: "sell", 0: "hold", 1: "buy"}
            signal_txt = signal_map.get(best_label, "hold")

            raw_conf = float(proba[best_idx]) * 100.0
            conf_margin = (
                float(proba[best_idx] - proba[second_idx]) * 100.0
                if len(order) > 1
                else raw_conf
            )

            probs_dict = {str(int(c)): float(proba[i]) for i, c in enumerate(classes)}

            return {
                "signal": signal_txt,
                "confidence": round(conf_margin, 2),
                "raw_conf": round(raw_conf, 2),
                "last_price": last_price_1m,
                "last_ts": last_ts_1m.isoformat(),  # UTC ISO
                "probs": probs_dict,
            }

        except Exception as e:
            # No reventar: devolver HOLD con error
            return {
                "signal": "hold",
                "confidence": 0.0,
                "raw_conf": 0.0,
                "last_price": last_price_1m,
                "last_ts": last_ts_1m.isoformat(),
                "error": f"predict_proba_failed: {e}",
            }

