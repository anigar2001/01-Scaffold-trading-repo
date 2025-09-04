# src/bot/models/multimodal_rl.py
from __future__ import annotations
import os, random, hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# SB3 opcional (inferir si hay .zip entrenado)
try:
    from stable_baselines3 import PPO
except Exception:
    PPO = None

DEFAULT_AGENT_PATH = os.getenv(
    "MULTIMODAL_RL_FILE",
    os.path.join(os.getenv("DATA_DIR", "/app/data"), "multimodal_rl.zip")
)

@dataclass
class ActOutput:
    symbols: List[str]
    action: List[float]                  # acción continua [-1,1] por símbolo
    equity_curve: Optional[List[float]]  # simulación corta (primer símbolo)
    positions_hist: Optional[List[int]]  # posiciones discretas (primer símbolo)
    meta: Dict[str, Any]

class MultimodalRLAgent:
    """
    - load(): intenta cargar PPO (SB3). Si no, queda en modo mock.
    - act(state): devuelve acción continua [-1,1] por símbolo y una simulación breve sobre el primero.
      state = {
        "symbols": ["BTCUSDT", ...],
        "price_window": [[...precios 1m...], ...],
        "sentiment": float | [float]
      }
    """
    def __init__(self, model_path: str = DEFAULT_AGENT_PATH):
        self.model_path = model_path
        self.agent: Optional[PPO] = None
        self.backend: Optional[str] = None
        # Deben coincidir con el entrenamiento
        self.lookback: int = int(os.getenv("RL_LOOKBACK", "32"))
        self.theta: float = float(os.getenv("RL_THETA", "0.05"))
        self.step_min: int = int(os.getenv("RL_STEP_MIN", "15"))  # agregación de 1m -> paso de 15m por defecto

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
        np.random.seed(h)

    @staticmethod
    def _extract_sent_for_index(sentiment: Any, i: int) -> float:
        if isinstance(sentiment, list):
            try:
                return float(sentiment[i] or 0.0)
            except Exception:
                return 0.0
        if isinstance(sentiment, (int, float)):
            return float(sentiment)
        return 0.0

    def _build_step_rets(self, prices_1m: List[float]) -> np.ndarray:
        """
        Deriva retornos log agregados a 'step_min' minutos a partir de precios 1m,
        y normaliza por 5*std (suave). Devuelve np.float32.
        """
        p = np.asarray(prices_1m, dtype=np.float64)
        if p.size < max(3, self.step_min + 1):
            return np.array([], dtype=np.float32)

        rets1m = np.diff(np.log(p + 1e-12))  # (N-1,)
        k = int(max(1, self.step_min))
        if rets1m.size < k:
            return np.array([], dtype=np.float32)

        # rolling-sum para paso de k minutos (e.g., 15m)
        kernel = np.ones(k, dtype=np.float64)
        r_step = np.convolve(rets1m, kernel, mode="valid").astype(np.float32)  # (N-1-(k-1),)

        # normalización suave
        std = float(np.std(r_step))
        scale = (5.0 * std + 1e-8) if std > 0 else 1.0
        return (r_step / scale).astype(np.float32)

    def _build_obs(self, rets_step_norm: np.ndarray, sentiment_val: float) -> np.ndarray:
        """Obs = últimos 'lookback' retornos agregados normalizados + 1 feature de sentimiento."""
        lb = int(self.lookback)
        if rets_step_norm.size < lb:
            pad = np.zeros(lb - rets_step_norm.size, dtype=np.float32)
            x = np.concatenate([pad, rets_step_norm[-rets_step_norm.size:]], dtype=np.float32)
        else:
            x = rets_step_norm[-lb:].astype(np.float32)
        obs = np.concatenate([x, np.array([float(sentiment_val)], dtype=np.float32)], dtype=np.float32)
        return obs  # shape = (lookback+1,)

    def _discretize(self, a: float) -> int:
        if abs(a) < self.theta:
            return 0
        return 1 if a > 0 else -1

    def _simulate_curve(
        self,
        rets_step_norm: np.ndarray,
        sentiment_val: float,
        use_sb3: bool
    ) -> Tuple[List[float], List[int], Dict[str, float]]:
        """
        Simula hasta 20 pasos con retornos agregados (e.g., 15m).
        """
        steps = min(20, max(0, rets_step_norm.size))
        if steps == 0:
            return [1.0], [], {"min": 0.0, "mean": 0.0, "max": 0.0}

        equity: List[float] = [1.0]
        pos_hist: List[int] = []
        a_hist: List[float] = []
        tail = rets_step_norm[-steps:]

        for k in range(steps):
            upto = rets_step_norm.size - steps + k + 1
            window = rets_step_norm[:upto]
            if use_sb3 and self.agent is not None:
                obs = self._build_obs(window, sentiment_val)
                a = float(self.agent.predict(obs, deterministic=True)[0].reshape(-1)[0])
                a = float(np.clip(a, -1.0, 1.0))
            else:
                base = float(np.clip(tail[k], -1.0, 1.0))
                a = 0.5 * base + 0.5 * float(sentiment_val) + 0.05 * (random.random() - 0.5)
                a = float(np.clip(a, -1.0, 1.0))

            pos = self._discretize(a)
            a_hist.append(a)
            pos_hist.append(pos)
            equity.append(equity[-1] * (1.0 + pos * float(tail[k])))

        a_hist_np = np.asarray(a_hist, dtype=np.float32)
        stats = {"min": float(a_hist_np.min()), "mean": float(a_hist_np.mean()), "max": float(a_hist_np.max())}
        return equity, pos_hist, stats

    def act(self, state: Dict[str, Any]) -> ActOutput:
        symbols = state.get("symbols") or ["BTCUSDT"]
        self._seed_from("|".join(map(str, symbols)))

        sentiment = state.get("sentiment", 0.0)
        price_window = state.get("price_window") or []

        actions: List[float] = []
        sim_equity: Optional[List[float]] = None
        sim_pos_hist: Optional[List[int]] = None
        sim_stats: Dict[str, float] = {}

        for i, _sym in enumerate(symbols):
            pw_i = price_window[i] if (isinstance(price_window, list) and i < len(price_window)) else []
            sent_i = self._extract_sent_for_index(sentiment, i)

            rets_step = self._build_step_rets(pw_i)  # retornos agregados (e.g., 15m) normalizados

            if self.backend == "sb3" and self.agent is not None:
                obs = self._build_obs(rets_step, sent_i)
                a = float(self.agent.predict(obs, deterministic=True)[0].reshape(-1)[0])
                a = float(np.clip(a, -1.0, 1.0))
            else:
                base = float(rets_step[-1]) if rets_step.size > 0 else 0.0
                a = 0.5 * base + 0.5 * sent_i + 0.05 * (random.random() - 0.5)
                a = float(np.clip(a, -1.0, 1.0))

            actions.append(a)

            if i == 0:
                use_sb3 = (self.backend == "sb3" and self.agent is not None)
                sim_equity, sim_pos_hist, sim_stats = self._simulate_curve(rets_step, sent_i, use_sb3)

        return ActOutput(
            symbols=symbols,
            action=actions,
            equity_curve=sim_equity,
            positions_hist=sim_pos_hist,
            meta={
                "backend": self.backend,
                "mock": self.backend is None,
                "theta": self.theta,
                "lookback": self.lookback,
                "step_min": self.step_min,
                "model_path": self.model_path,
                "obs_dim": self.lookback + 1,
                "scaling": "returns_step_norm",
                "action_stats": sim_stats,
            },
        )
