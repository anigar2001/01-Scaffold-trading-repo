# src/ai_training/train_multimodal.py
from __future__ import annotations
import os
import argparse
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

# -------------------------------------------------
# Datos
# -------------------------------------------------
def _load_ohlcv(symbol: str, tf: str = "1m") -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Devuelve (closes, timestamps_ns) para el símbolo/TF. Cierra si faltan datos.
    """
    import pandas as pd
    path = os.path.join(DATA_DIR, f"ohlcv_{symbol}_{tf}.csv")
    if not os.path.exists(path):
        return None, None
    df = pd.read_csv(path)
    if "close" not in df.columns or df.empty:
        return None, None
    ts = None
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
        ts = df["timestamp"].to_numpy(dtype="datetime64[ns]")
    closes = pd.to_numeric(df["close"], errors="coerce").dropna().to_numpy(dtype=np.float64)
    if closes.size < 2000:
        return None, None
    return closes, ts

def _load_sentiment(symbol: str) -> Optional[Dict[str, float]]:
    """
    Lee /app/data/sentiment_{SYMBOL}.csv con columnas: date, sentiment
    Devuelve dict {YYYY-MM-DD: value}
    """
    import pandas as pd
    path = os.path.join(DATA_DIR, f"sentiment_{symbol}.csv")
    if not os.path.exists(path):
        # fallback para BTCUSDT
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

def _align_sentiment_daily_to_prices(closes: np.ndarray, timestamps: Optional[np.ndarray],
                                     sent_map: Optional[Dict[str, float]]) -> np.ndarray:
    """
    Alinea sentimiento DIARIO a cada cierre (mismo len que closes).
    Si no hay timestamps o no hay map, devuelve ceros.
    """
    if sent_map is None or timestamps is None:
        return np.zeros_like(closes, dtype=np.float32)
    dates = np.array([str(np.datetime_as_string(ts, unit="D")) for ts in timestamps])
    return np.array([float(sent_map.get(d, 0.0)) for d in dates], dtype=np.float32)

def _make_step_returns_and_sent(
    closes_1m: np.ndarray,
    sent_daily_per_close: np.ndarray,
    step_min: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    A partir de precios 1m:
      - r_step_raw: retornos log agregados a 'step_min' (len = N-1-(k-1))
      - r_step_norm: normalizados por 5*std
      - sent_step: sentimiento alineado a esos pasos (tomando el valor del último minuto del bloque)
    """
    # rets 1m
    rets1m = np.diff(np.log(closes_1m + 1e-12))  # (N-1,)
    k = int(max(1, step_min))
    if rets1m.size < k:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32), np.array([], dtype=np.float32)

    # rolling-sum para k minutos
    kernel = np.ones(k, dtype=np.float64)
    r_step_raw = np.convolve(rets1m, kernel, mode="valid").astype(np.float32)  # (N-1-(k-1),)

    # normalización suave
    std = float(np.std(r_step_raw))
    scale = (5.0 * std + 1e-8) if std > 0 else 1.0
    r_step_norm = (r_step_raw / scale).astype(np.float32)

    # sentimiento 1m alineado con rets1m es sent_daily_per_close[1:]
    if sent_daily_per_close.size >= 2:
        sent1m = sent_daily_per_close[1:].astype(np.float32)
    else:
        sent1m = np.zeros_like(rets1m, dtype=np.float32)

    # para cada ventana k, tomamos el sentimiento del último minuto del bloque
    if sent1m.size >= k:
        sent_step = sent1m[k - 1:].astype(np.float32)       # len = len(r_step_raw)
    else:
        sent_step = np.zeros_like(r_step_raw, dtype=np.float32)

    return r_step_raw, r_step_norm, sent_step

# -------------------------------------------------
# Entorno Gymnasium (paso = step_min, obs = lookback de r_step_norm + sentiment)
# -------------------------------------------------
def make_trading_env(symbol: str, lookback: int, steps_per_ep: int, step_min: int, seed: int = 42):
    assert gym is not None

    closes, ts = _load_ohlcv(symbol, "1m")
    if closes is None:
        # sintético para no romper
        rng = np.random.default_rng(123)
        n = 20_000
        rets = rng.normal(0.0, 0.0008, size=n)
        closes = 25_000 * np.exp(np.cumsum(rets))
        ts = None

    sent_map = _load_sentiment(symbol)
    sent_daily_per_close = _align_sentiment_daily_to_prices(closes, ts, sent_map)

    r_step_raw, r_step_norm, sent_step = _make_step_returns_and_sent(closes, sent_daily_per_close, step_min)

    class NewsPriceEnv(gym.Env):
        metadata = {"render_modes": []}
        def __init__(self):
            super().__init__()
            self.lookback = int(lookback)
            self.steps_per_ep = int(steps_per_ep)
            self.pos = 0.0
            self.i0 = 0
            self.i = 0

            self.r_raw = r_step_raw
            self.r_n = r_step_norm
            self.sen = sent_step
            self.M = int(self.r_n.shape[0])
            # margen para slicing y episodios
            self.max_i = max(0, self.M - 1)

            obs_dim = self.lookback + 1
            self.observation_space = gym.spaces.Box(low=-5, high=5, shape=(obs_dim,), dtype=np.float32)
            self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

            self._rng = np.random.default_rng(seed)

        def _get_obs(self) -> np.ndarray:
            # ventana de retornos normalizados [i-L, i) y sentimiento del último punto
            sl = slice(self.i - self.lookback, self.i)
            x = self.r_n[sl]
            if x.size < self.lookback:
                pad = np.zeros(self.lookback - x.size, dtype=np.float32)
                x = np.concatenate([pad, x], dtype=np.float32)
            s = self.sen[self.i - 1] if self.i - 1 >= 0 else 0.0
            return np.concatenate([x, np.array([s], dtype=np.float32)], dtype=np.float32)

        def reset(self, *, seed=None, options=None):
            if seed is not None:
                self._rng = np.random.default_rng(seed)
            # elige inicio con holgura para lookback y duración
            start_min = self.lookback + 2
            start_max = max(start_min + 1, self.max_i - self.steps_per_ep - 2)
            if start_max <= start_min:
                self.i0 = start_min
            else:
                self.i0 = int(self._rng.integers(start_min, start_max))
            self.i = self.i0
            self.pos = 0.0
            obs = self._get_obs()
            return obs, {}

        def step(self, action: np.ndarray):
            a = float(np.clip(action[0], -1.0, 1.0))
            # recompensa con retorno "raw" del paso actual y pequeño coste
            r_next = float(self.r_raw[self.i]) if self.i < self.M else 0.0
            trans_cost = 0.0000 * abs(a - self.pos)  # empieza en 0.0 para CPU; súbelo cuando estabilice
            reward = a * r_next - trans_cost

            self.pos = a
            self.i += 1
            terminated = (self.i - self.i0) >= self.steps_per_ep or self.i >= self.M
            truncated = False
            obs = self._get_obs()
            info: Dict[str, Any] = {}
            return obs, reward, terminated, truncated, info

    return NewsPriceEnv

# -------------------------------------------------
# Entrenamiento
# -------------------------------------------------
def train(symbol: str, timesteps: int, lookback: int, steps_per_ep: int, n_envs: int, step_min: int, seed: int) -> None:
    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)

    if not _HAS_SB3:
        # Fallback dummy si no hay SB3/Gym
        with open(OUT_FILE, "w") as f:
            f.write("DUMMY_RL\n")
        print(f"[OK] Guardado dummy RL en {OUT_FILE} (sin SB3).")
        return

    EnvCls = make_trading_env(symbol=symbol, lookback=lookback, steps_per_ep=steps_per_ep, step_min=step_min, seed=seed)
    vec_env = make_vec_env(EnvCls, n_envs=n_envs, seed=seed)

    policy_kwargs = dict(net_arch=[128, 128])
    model = PPO("MlpPolicy", vec_env, verbose=1, seed=seed, tensorboard_log=TB_LOG_DIR, policy_kwargs=policy_kwargs)

    print(f"[INFO] Entrenando RL: symbol={symbol} step_min={step_min} timesteps={timesteps} lookback={lookback} steps/ep={steps_per_ep} n_envs={n_envs}")
    model.learn(total_timesteps=int(timesteps), progress_bar=False)

    model.save(OUT_FILE)
    print(f"[OK] Guardado agente RL en {OUT_FILE}")

# -------------------------------------------------
# CLI
# -------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Entrena agente RL multimodal (precio + sentimiento) con paso agregado (p.ej., 15m).")
    p.add_argument("--symbol", type=str, default="BTCUSDT", help="Símbolo (default: BTCUSDT)")
    p.add_argument("--timesteps", type=int, default=200_000, help="Total timesteps de entrenamiento")
    p.add_argument("--lookback", type=int, default=int(os.getenv("RL_LOOKBACK", "32")), help="Ventana de retornos agregados en la observación")
    p.add_argument("--steps-per-episode", type=int, default=256, help="Pasos por episodio")
    p.add_argument("--n-envs", type=int, default=1, help="N entornos vectorizados (1 en CPU está bien)")
    p.add_argument("--step-min", type=int, default=int(os.getenv("RL_STEP_MIN", "15")), help="Minutos por paso (agregación desde 1m)")
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
        step_min=args.step_min,
        seed=args.seed,
    )
