# src/bot/models/multimodal_rl.py
from __future__ import annotations
import os, random, hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

try:
    from stable_baselines3 import PPO  # opcional
except Exception:
    PPO = None

DEFAULT_AGENT_PATH = os.getenv(
    "MULTIMODAL_RL_FILE",
    os.path.join(os.getenv("DATA_DIR", "/app/data"), "multimodal_rl.zip")
)

@dataclass
class ActOutput:
    symbols: List[str]
    action: List[float]                  # [-1,1] por símbolo
    equity_curve: Optional[List[float]]
    positions_hist: Optional[List[int]]
    meta: Dict[str, Any]

class MultimodalRLAgent:
    """
    - load(): intenta cargar PPO (SB3). Si no, queda mock.
    - act(state): acción continua [-1,1] por símbolo.
    """
    def __init__(self, model_path: str = DEFAULT_AGENT_PATH):
        self.model_path = model_path
        self.agent = None
        self.backend = None

    def load(self) -> bool:
        if PPO is not None and os.path.exists(self.model_path):
            try:
                self.agent = PPO.load(self.model_path)
                self.backend = "sb3"
                return True
            except Exception:
                pass
        self.agent, self.backend = None, None
        return False

    def _seed_from(self, key: str) -> None:
        h = int(hashlib.sha256(key.encode("utf-8")).hexdigest(), 16) % (2**32)
        random.seed(h)

    def act(self, state: Dict[str, Any]) -> ActOutput:
        symbols = state.get("symbols") or ["BTCUSDT"]
        self._seed_from("|".join(symbols))

        # TODO si self.backend == "sb3": construir obs real y predecir
        # MOCK: momentum + sentimiento
        actions: List[float] = []
        for i, _ in enumerate(symbols):
            base = 0.0
            pw = state.get("price_window") or []
            if pw and i < len(pw) and len(pw[i]) >= 2:
                last, prev = float(pw[i][-1]), float(pw[i][-2])
                if prev != 0:
                    base += max(-1.0, min(1.0, (last - prev) / abs(prev)))
            sent = 0.0
            sentiment = state.get("sentiment")
            if isinstance(sentiment, list) and i < len(sentiment):
                sent = float(sentiment[i] or 0.0)
            elif isinstance(sentiment, (int, float)):
                sent = float(sentiment)
            act = max(-1.0, min(1.0, 0.5 * base + 0.5 * sent + 0.05 * (random.random() - 0.5)))
            actions.append(act)

        equity = [1.0]
        pos_hist = []
        for _ in range(20):
            step = actions[0] * (0.002 + 0.001 * random.random())
            equity.append(equity[-1] * (1.0 + step))
            pos_hist.append(-1 if actions[0] < -0.33 else (1 if actions[0] > 0.33 else 0))

        return ActOutput(
            symbols=symbols,
            action=actions,
            equity_curve=equity,
            positions_hist=pos_hist,
            meta={"backend": self.backend, "mock": self.backend is None},
        )
