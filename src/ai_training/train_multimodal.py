# src/ai_training/train_multimodal.py
from __future__ import annotations
import os
import argparse
import math
from typing import Dict, Any, Tuple, Optional

import numpy as np

# SB3 / Gym (opcionales). Si no están, guardamos dummy.
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    import gymnasium as gym
    _HAS_SB3 = True
except ImportError:
    PPO = None
    make_vec_env = None
    gym = None
    _HAS_SB3 = False

DATA_DIR = os.getenv("DATA_DIR", "/app/data")
OUT_FILE = os.getenv("MULTIMODAL_RL_FILE", os.path.join(DATA_DIR, "multimodal_rl.zip"))
TB_LOG_DIR = os.getenv("TB_LOG_DIR", os.path.join(DATA_DIR, "tb_logs"))

# ----------------------------
# Utilidades de datos
# ----------------------------
def _load_ohlcv(symbol: str, tf: str = "1m") -> Optional[np.ndarray]:
    """
    Devuelve array con la columna 'close' ordenada por tiempo (float64).
    Busca /app/data/ohlcv_{SYMBOL}_{tf}.csv
    """
    import pandas as pd
    path = os.path.join(DATA_DIR, f"ohlcv_{symbol}_{tf}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if "close" not in df.columns or df.empty:
        return None
    # Asegura orden temporal
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    closes = pd.to_numeric(df["close"], errors="coerce").dropna().to_numpy(dtype=np.float64)
    if closes.size < 1000:
        return None
    return closes

def _load_sentiment(symbol: str) -> Optional[Dict[str, float]]:
    """
    Carga sentimiento diario. Devuelve dict {YYYY-MM-DD: value}.
    Busca /app/data/sentiment_{SYMBOL}.csv (e.g., sentiment_BTCUSDT.csv).
    """
    import pandas as pd
    path = os.path.join(DATA_DIR, f"sentiment_{symbol}.csv")
    if not os.path.exists(path):
        # intenta nombre por defecto para BTCUSDT
        if symbol.upper() == "BTCUSDT":
            path2 = os.path.join(DATA_DIR, "sentiment_BTCUSDT.csv")
            if os.path.exists(path2):
                path = path2
            else:
                return None
        else:
            return None
    df = pd.read_csv(path)
    if "date" not in df.columns or "sentiment" not in df.columns or df.empty:
        return None
    df = df.dropna(subset=["date", "sentiment"])
    try:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date.astype(str)
    except Exception:
        df["date"] = df["date"].astype(str)
    return dict(zip(df["date"].astype(str), df["sentiment"].astype(float)))

def _align_sentiment_to_closes(closes: np.ndarray, timestamps: Optional[np.ndarray], sent_map: Optional[Dict[str, float]]) -> np.ndarray:
    """
    Alinea sentimiento diario a cada barra (por fecha). Si no hay, devuelve ceros.
    timestamps: serie de np.datetime64[ns] o None (si no la tenemos).
    """
    if sent_map is None or timestamps is None:
        return np.zeros_like(closes, dtype=np.float32)
    # Extrae fecha 'YYYY-MM-DD' por barra
    dates = np.array([str(np.datetime_as_string(ts, unit="D")) for ts in timestamps])
    out = np.array([float(sent_map.get(d, 0.0)) for d in dates], dtype=np.float32)
    return out

def _load_data_for_env(symbol: str, tf: str = "1m") -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """
    Devuelve (closes, timestamps, sentiment_series_alineada)
    Si no hay datos reales, genera sintéticos.
    """
    import pandas as pd
    path = os.path.join(DATA_DIR, f"ohlcv_{symbol}_{tf}.csv")
    closes = None
    timestamps = None
    if os.path.exists(path):
        df = pd.read_csv(path)
        if "close" in df.columns:
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
                df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
                timestamps = df["timestamp"].to_numpy(dtype="datetime64[ns]")
            closes = pd.to_numeric(df["close"], errors="coerce").dropna().to_numpy(dtype=np.float64)

    if closes is None or closes.size < 2000:
        # Sintético: random walk (para que el pipeline funcione sin datos)
        rng = np.random.default_rng(123)
        n = 20_000
        rets = rng.normal(loc=0.0, scale=0.0008, size=n)
        closes = 25_000 * np.exp(np.cumsum(rets))
        timestamps = None

    sent_map = _load_sentiment(symbol)
    sent_series = _align_sentiment_to_closes(closes, timestamps, sent_map)
    return closes, timestamps, sent_series

# ----------------------------
# Entorno Gymnasium
# ----------------------------
def make_trading_env(symbol: str, lookback: int, steps_per_ep: int, seed: int = 42):
    """
    Env continuo con acción a_t in [-1,1] (posición).
    Observación: [ret_{t-L+1}..ret_t, sentiment_t]  => dim = lookback + 1
    Recompensa:  r_{t+1} * posición_actual
    """
    assert gym is not None

    closes, _, sent_series = _load_data_for_env(symbol, tf="1m")
    # Retornos logarítmicos
    logp = np.log(closes + 1e-12)
    rets = np.diff(logp).astype(np.float32)
    sent = sent_series[1:].astype(np.float32)  # alinea con rets

    # Normalización simple de retornos
    std = np.std(rets)
    if std > 0:
        rets_n = rets / (5 * std)  # suaviza
    else:
        rets_n = rets

    class NewsPriceEnv(gym.Env):
        metadata = {"render_modes": []}
        def __init__(self):
            super().__init__()
            self.lookback = int(lookback)
            self.steps_per_ep = int(steps_per_ep)
            self.pos = 0.0
            self.i0 = 0
            self.i = 0
            self.max_i = rets_n.shape[0] - self.lookback - 2

            obs_dim = self.lookback + 1
            self.observation_space = gym.spaces.Box(low=-5, high=5, shape=(obs_dim,), dtype=np.float32)
            self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

            # seeds
            self._rng = np.random.default_rng(seed)

        def _get_obs(self) -> np.ndarray:
            sl = slice(self.i - self.lookback, self.i)
            obs = np.concatenate([rets_n[sl], sent[sl][-1:]], dtype=np.float32)
            return obs

        def reset(self, *, seed=None, options=None):
            if seed is not None:
                self._rng = np.random.default_rng(seed)
            # Selecciona un inicio aleatorio con holgura
            self.i0 = int(self._rng.integers(self.lookback + 2, self.max_i - self.steps_per_ep - 2))
            self.i = self.i0
            self.pos = 0.0
            obs = self._get_obs()
            return obs, {}

        def step(self, action: np.ndarray):
            a = float(np.clip(action[0], -1.0, 1.0))
            # recompensa por retorno siguiente con slippage/penalización leve por cambio de posición
            r_next = float(rets[self.i + 1])
            trans_cost = 0.00005 * abs(a - self.pos)
            reward = a * r_next - trans_cost

            self.pos = a
            self.i += 1
            terminated = (self.i - self.i0) >= self.steps_per_ep
            truncated = False
            obs = self._get_obs()
            info: Dict[str, Any] = {}
            return obs, reward, terminated, truncated, info

    return NewsPriceEnv

# ----------------------------
# Entrenamiento
# ----------------------------
def train(symbol: str, timesteps: int, lookback: int, steps_per_ep: int, n_envs: int, seed: int) -> None:
    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)

    if not _HAS_SB3:
        # Fallback dummy si no hay SB3/Gym
        with open(OUT_FILE, "w") as f:
            f.write("DUMMY_RL\n")
        print(f"[OK] Guardado dummy RL en {OUT_FILE} (sin SB3).")
        return

    EnvCls = make_trading_env(symbol=symbol, lookback=lookback, steps_per_ep=steps_per_ep, seed=seed)
    # Vec env
    vec_env = make_vec_env(EnvCls, n_envs=n_envs, seed=seed)

    policy_kwargs = dict(net_arch=[128, 128])
    model = PPO("MlpPolicy", vec_env, verbose=1, seed=seed, tensorboard_log=TB_LOG_DIR, policy_kwargs=policy_kwargs)

    print(f"[INFO] Entrenando RL: symbol={symbol} timesteps={timesteps} lookback={lookback} steps/ep={steps_per_ep} n_envs={n_envs}")
    model.learn(total_timesteps=int(timesteps), progress_bar=False)

    # Guarda en OUT_FILE (zip SB3)
    model.save(OUT_FILE)
    print(f"[OK] Guardado agente RL en {OUT_FILE}")

# ----------------------------
# CLI
# ----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Entrena agente RL multimodal (precio + sentimiento).")
    p.add_argument("--symbol", type=str, default="BTCUSDT", help="Símbolo (default: BTCUSDT)")
    p.add_argument("--timesteps", type=int, default=200_000, help="Total timesteps de entrenamiento")
    p.add_argument("--lookback", type=int, default=32, help="Ventana de retornos en la observación")
    p.add_argument("--steps-per-episode", type=int, default=512, help="Pasos por episodio")
    p.add_argument("--n-envs", type=int, default=1, help="N entornos vectorizados (1 en CPU está bien)")
    p.add_argument("--seed", type=int, default=42, help="Semilla")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train(
        symbol=args.symbol,
        timesteps=args.timesteps,
        lookback=args.lookback,
        steps_per_ep=args.steps_per_episode,
        n_envs=args.n_envs,
        seed=args.seed,
    )

