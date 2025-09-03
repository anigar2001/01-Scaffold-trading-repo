# src/bot/models/stockformer.py
from __future__ import annotations
import os, random, hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

try:
    import torch  # opcional
except Exception:
    torch = None

DEFAULT_MODEL_PATH = os.getenv(
    "STOCKFORMER_MODEL_FILE",
    os.path.join(os.getenv("DATA_DIR", "/app/data"), "stockformer_model.pt")
)

@dataclass
class PredictOutput:
    symbol: str
    probs: Dict[str, float]              # {"buy":x,"hold":y,"sell":z}
    attention_by_tf: Dict[str, float]    # {"1m":..., "5m":..., "15m":..., "1h":...}
    meta: Dict[str, Any]

class StockformerModel:
    """
    - load(): intenta cargar pesos (torch). Si no, queda en modo mock.
    - predict(features): devuelve probs buy/hold/sell + atención por timeframe.
    """
    def __init__(self, model_path: str = DEFAULT_MODEL_PATH):
        self.model_path = model_path
        self.model = None
        self.backend = None

    def load(self) -> bool:
        if torch is not None and os.path.exists(self.model_path):
            try:
                self.model = torch.jit.load(self.model_path)
                self.model.eval()
                self.backend = "torch"
                return True
            except Exception:
                pass
        self.model = None
        self.backend = None
        return False

    def _seed_from(self, key: str) -> None:
        h = int(hashlib.sha256(key.encode("utf-8")).hexdigest(), 16) % (2**32)
        random.seed(h)

    def predict(self, features: Dict[str, Any]) -> PredictOutput:
        # Espera: {"symbol": "BTCUSDT", "timeframes": [...], "features": {...}}
        symbol = (features.get("symbol") or "UNKNOWN").upper()
        tfs: List[str] = features.get("timeframes") or list((features.get("features") or {}).keys()) or ["1m","5m","15m","1h"]

        # TODO si self.backend == "torch": construir tensores y llamar al modelo real
        # MOCK determinista por (symbol,tfs)
        self._seed_from(symbol + "|" + "|".join(sorted(tfs)))
        raw = [random.random() + 1e-6 for _ in tfs]
        s = sum(raw)
        attention = {tf: v/s for tf, v in zip(tfs, raw)}

        # Heurística: TF rápidas → más "buy"; lentas → más "sell"
        fast_weight = sum(attention.get(tf, 0.0) for tf in ("1m","5m"))
        buy = 0.55 * fast_weight + 0.15 * random.random()
        sell = 0.35 * (1.0 - fast_weight) + 0.15 * random.random()
        hold = max(0.0, 1.0 - buy - sell)
        norm = buy + hold + sell
        probs = {"buy": buy/norm, "hold": hold/norm, "sell": sell/norm}

        return PredictOutput(
            symbol=symbol,
            probs=probs,
            attention_by_tf=attention,
            meta={"backend": self.backend, "mock": self.backend is None},
        )
