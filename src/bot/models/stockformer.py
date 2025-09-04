# src/bot/models/stockformer.py
from __future__ import annotations

import os
import random
import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import torch  # opcional
except Exception:
    torch = None

DEFAULT_MODEL_PATH = os.getenv(
    "STOCKFORMER_MODEL_FILE",
    os.path.join(os.getenv("DATA_DIR", "/app/data"), "stockformer_model.pt"),
)


@dataclass
class PredictOutput:
    symbol: str
    probs: Dict[str, float]              # {"buy":x,"hold":y,"sell":z}
    attention_by_tf: Dict[str, float]    # {"1m":..., "5m":..., "15m":..., "1h":...}
    meta: Dict[str, Any]


class StockformerModel:
    """
    Interfaz del modelo de clasificación 15m (buy/hold/sell).

    Métodos:
      - load(): intenta cargar pesos (torch). Si no, queda en modo mock/heurístico.
      - predict(features): compat histórica -> devuelve PredictOutput.
      - predict_from_features(features, horizon_min=15): devuelve dict {'buy','hold','sell'}.
      - predict_from_ohlcv(dfs, horizon_min=15): idem, construyendo features desde OHLCV multi-TF.
    """
    def __init__(self, model_path: str = DEFAULT_MODEL_PATH):
        self.model_path = model_path
        self.model = None
        self.backend: Optional[str] = None  # "torch" o None

    # ---------------------------
    # Carga
    # ---------------------------
    def load(self) -> bool:
        if torch is not None and os.path.exists(self.model_path):
            try:
                # Permitimos que sea torchscript o un state_dict envuelto
                self.model = torch.jit.load(self.model_path)
                self.model.eval()
                self.backend = "torch"
                return True
            except Exception:
                # Si no es TorchScript, intentamos torch.load genérico
                try:
                    self.model = torch.load(self.model_path, map_location="cpu")
                    # Si tiene .eval(), lo llamamos
                    if hasattr(self.model, "eval"):
                        self.model.eval()
                    self.backend = "torch"
                    return True
                except Exception:
                    pass
        self.model = None
        self.backend = None
        return False

    # ---------------------------
    # Utils internos
    # ---------------------------
    def _seed_from(self, key: str) -> None:
        h = int(hashlib.sha256(key.encode("utf-8")).hexdigest(), 16) % (2**32)
        random.seed(h)

    @staticmethod
    def _safe_softmax(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        x = x - np.max(x)
        ex = np.exp(x)
        s = ex.sum()
        return ex / s if s > 0 else np.ones_like(ex) / len(ex)

    @staticmethod
    def _normalize_probs(buy: float, hold: float, sell: float) -> Dict[str, float]:
        v = np.array([sell, hold, buy], dtype=np.float64)
        v = np.clip(v, 1e-9, None)
        v = v / v.sum()
        return {"buy": float(v[2]), "hold": float(v[1]), "sell": float(v[0])}

    # ---------------------------
    # Heurística determinista (sin Torch)
    # ---------------------------
    def _heuristic_from_mtf_mom(self, mtf_mom: Dict[str, float]) -> Dict[str, float]:
        """
        mtf_mom: momentum por TF (media/std últimos n), e.g. {'1m': 0.1, '5m': -0.05, ...}
        Devuelve probs buy/hold/sell en base a momentum ponderado por 'atención' por TF.
        """
        tfs = [tf for tf in ["1m", "5m", "15m", "1h"] if tf in mtf_mom]
        if not tfs:
            # Sin datos: equilibrio
            return {"buy": 1/3, "hold": 1/3, "sell": 1/3}

        # Atención: softmax del |momentum| (lo más informativo pesa más)
        att_raw = np.array([abs(mtf_mom[tf]) for tf in tfs], dtype=np.float64)
        if not np.isfinite(att_raw).all() or att_raw.sum() <= 0:
            att = np.ones(len(tfs), dtype=np.float64) / len(tfs)
        else:
            att = self._safe_softmax(att_raw)

        # Señal agregada = sum(momentum * atención)
        s = float(np.sum(np.array([mtf_mom[tf] for tf in tfs]) * att))

        # Mapear señal -> probs (tanh para limitar)
        bias = float(np.tanh(2.0 * s))  # [-1,1]
        p_buy = 0.33 + 0.33 * max(0.0, bias)
        p_sell = 0.33 + 0.33 * max(0.0, -bias)
        p_hold = max(1e-9, 1.0 - (p_buy + p_sell))
        return self._normalize_probs(p_buy, p_hold, p_sell)

    # ---------------------------
    # API pública
    # ---------------------------
    def predict(self, features: Dict[str, Any]) -> PredictOutput:
        """
        Compat histórica (no romper llamadas existentes del dashboard).
        Espera:
          features = {
            "symbol": "BTCUSDT",
            "timeframes": ["1m","5m","15m","1h"],
            "features": {...}  # opcional
          }
        Si self.backend == "torch" y en el futuro conectamos tensores, aquí iría la llamada real.
        De momento, produce una predicción determinista (mock/heurística).
        """
        symbol = (features.get("symbol") or "UNKNOWN").upper()
        tfs: List[str] = features.get("timeframes") or list((features.get("features") or {}).keys()) \
                         or ["1m", "5m", "15m", "1h"]

        # MOCK determinista por (symbol, TFs)
        self._seed_from(symbol + "|" + "|".join(sorted(tfs)))
        raw = [random.random() + 1e-6 for _ in tfs]
        s = sum(raw)
        attention = {tf: v / s for tf, v in zip(tfs, raw)}

        # Heurística simple: TF rápidas → más "buy"; lentas → más "sell"
        fast_weight = sum(attention.get(tf, 0.0) for tf in ("1m", "5m"))
        buy = 0.55 * fast_weight + 0.15 * random.random()
        sell = 0.35 * (1.0 - fast_weight) + 0.15 * random.random()
        hold = max(0.0, 1.0 - buy - sell)
        probs = self._normalize_probs(buy, hold, sell)

        return PredictOutput(
            symbol=symbol,
            probs=probs,
            attention_by_tf=attention,
            meta={"backend": self.backend, "mock": self.backend is None},
        )

    def predict_from_features(self, features: Dict[str, Any], horizon_min: int = 15) -> Dict[str, float]:
        """
        Versión que devuelve un dict {'buy','hold','sell'} (lo que espera la API).
        Si en el futuro conectamos Torch real, este es el lugar para generar el tensor x y llamar al modelo.
        """
        # Por ahora reusamos la compat .predict() y extraemos probs
        out = self.predict(features)
        return {
            "buy": float(out.probs.get("buy", 1/3)),
            "hold": float(out.probs.get("hold", 1/3)),
            "sell": float(out.probs.get("sell", 1/3)),
        }

    def predict_from_ohlcv(self, dfs: Dict[str, "pd.DataFrame"], horizon_min: int = 15) -> Dict[str, float]:
        """
        Construye una predicción a partir de OHLCV multi-timeframe.
        dfs: dict { "1m": df1m, "5m": df5m, "15m": df15m, "1h": df1h }
             Cada df con columnas: ['timestamp','open','high','low','close','volume'].

        Heurística estable (si Torch no está integrado): momentum z-normalizado por TF.
        """
        # Import local para evitar dependencia dura si sólo usas .predict()
        import pandas as pd  # type: ignore

        if not isinstance(dfs, dict) or not dfs:
            # Sin datos -> equilibrio
            return {"buy": 1/3, "hold": 1/3, "sell": 1/3}

        # Lookback por TF (últimas velas a considerar en returns)
        # Ajusta si lo deseas; números pequeños para ser ligeros.
        LB = {"1m": 128, "5m": 96, "15m": 64, "1h": 32}

        mtf_mom: Dict[str, float] = {}
        for tf, df in dfs.items():
            if tf not in ("1m", "5m", "15m", "1h"):
                continue
            if df is None or len(df) < 10:
                continue

            # Tolerancia de nombres de columnas
            cols = {c.lower() for c in df.columns}
            if not {"close"}.issubset(cols):
                # intentar renombrar si viene en otro formato
                df = df.rename(columns={c: c.lower() for c in df.columns})
                if "close" not in df.columns:
                    continue

            # Últimas N velas por TF
            n = LB.get(tf, 64)
            c = pd.to_numeric(df["close"], errors="coerce").dropna().to_numpy(dtype=np.float64)[-max(16, n):]
            if c.size < 16:
                continue
            r = np.diff(np.log(c))
            # Momentum z-normalizado
            mu = float(np.mean(r[-n//2:]))  # media de la mitad más reciente
            sd = float(np.std(r[-n:])) + 1e-8
            mtf_mom[tf] = mu / sd

        if not mtf_mom:
            # Sin TF válidos -> equilibrio
            return {"buy": 1/3, "hold": 1/3, "sell": 1/3}

        # Si tuviéramos integración Torch real, aquí construiríamos el tensor de entrada
        # en base a retornos/indicadores y llamaríamos a self.model(...).
        # try:
        #     if self.backend == "torch" and self.model is not None:
        #         x = ...  # construir tensores a partir de dfs
        #         with torch.no_grad():
        #             logits = self.model(x)  # -> tensor [1,3]
        #         p = torch.softmax(logits, dim=-1).cpu().numpy().ravel()
        #         return {"buy": float(p[2]), "hold": float(p[1]), "sell": float(p[0])}
        # except Exception:
        #     pass

        # Heurística determinista por TF (estable y suficiente para la API)
        return self._heuristic_from_mtf_mom(mtf_mom)
