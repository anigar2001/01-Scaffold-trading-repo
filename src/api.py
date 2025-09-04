# src/api.py
# -*- coding: utf-8 -*-
"""
API del bot de trading (FastAPI)

- Señales:
  * POST /signal/stockformer                 -> probabilidades buy/hold/sell a 15m (con fallback seguro)
  * POST /agent/multimodal/action           -> acción continua [-1,+1] + positions_hist + equity_curve (con fallback)

- Modelos:
  * POST /models/reload?target=stockformer|multimodal|all

- Entrenamiento (lanzamiento en background, opcional):
  * POST /train/stockformer
  * POST /train/multimodal
  * GET  /train/jobs
  * GET  /train/jobs/{job_id}
  * GET  /train/logs/{job_id}

- Ingesta / Sentimiento (opcional):
  * POST /ingest/news
  * POST /build/sentiment

Notas:
- Compatibles con Pydantic v1 (.dict()).
- No se crean archivos nuevos en el repo; todo va aquí y usa rutas/ENV existentes.
"""

import os
import csv
import json
import uuid
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Cargar .env
load_dotenv()

# -----------------------
# Conectores y servicios
# -----------------------
from connectors.ccxt_connector import CCXTConnector
from connectors.mock_exchange import MockExchange
from order_manager import OrderManager
from bot.ai_signal import AISignal
from utils.logger import LOG_FILE

# Modelos IA enchufables (nuevos)
from bot.models.stockformer import StockformerModel
from bot.models.multimodal_rl import MultimodalRLAgent

# Construir sentimientos (opcionales)
from data_pipeline.ingest_news import ingest_sources
from ai_training.build_sentiment import build_sentiment
from data_pipeline.news_sources import DEFAULT_SYMBOL, SENTIMENT_CSV, DRIVERS_CSV

# -----------------------
# Configuración conector
# -----------------------
CONNECTOR_TYPE = os.getenv("CONNECTOR_TYPE", "MOCK").upper()
if CONNECTOR_TYPE == "CCXT":
    connector = CCXTConnector()
else:
    connector = MockExchange()

order_manager = OrderManager(connector)
app = FastAPI(title="Trading Bot API")

# -----------------------
# Singletons (lazy)
# -----------------------
_ai_signal: Optional[AISignal] = None
_stockformer: Optional[StockformerModel] = None
_rl_agent: Optional[MultimodalRLAgent] = None


def get_ai() -> Tuple[Optional[AISignal], Optional[str]]:
    """Devuelve instancia AISignal o (None, error) si no se puede cargar."""
    global _ai_signal
    if _ai_signal is not None:
        return _ai_signal, None
    try:
        _ai_signal = AISignal(connector=connector)
        return _ai_signal, None
    except FileNotFoundError as e:
        return None, str(e)
    except Exception as e:
        return None, f"No se pudo inicializar AISignal: {e}"


def get_stockformer() -> StockformerModel:
    global _stockformer
    if _stockformer is None:
        _stockformer = StockformerModel()
        _stockformer.load()
    return _stockformer


def get_rl_agent() -> MultimodalRLAgent:
    global _rl_agent
    if _rl_agent is None:
        _rl_agent = MultimodalRLAgent()
        _rl_agent.load()
    return _rl_agent


# -----------------------
# Constantes / helpers
# -----------------------
ALLOWED_TFS = {"1m", "5m", "15m", "1h"}

# ENV datos/modelos
DATA_DIR = Path(os.getenv("DATA_DIR", "/app/data"))
MODELS_DIR = Path(os.getenv("MODELS_DIR", "/app/models"))  # opcional
LOGS_DIR = Path(os.getenv("LOGS_DIR", "/app/logs"))
JOBS_DIR = DATA_DIR / "train_jobs"

# ENV Stockformer
STOCKFORMER_HORIZON_MIN = int(os.getenv("STOCKFORMER_HORIZON_MIN", 15))
STOCKFORMER_MODEL_FILE = os.getenv(
    "STOCKFORMER_MODEL_FILE", str(DATA_DIR / "stockformer_model.pt")
)

# ENV RL
RL_STEP_MIN = int(os.getenv("RL_STEP_MIN", 15))
RL_LOOKBACK = int(os.getenv("RL_LOOKBACK", 32))
RL_THETA = float(os.getenv("RL_THETA", 0.05))
MULTIMODAL_RL_FILE = os.getenv(
    "MULTIMODAL_RL_FILE", str(DATA_DIR / "multimodal_rl.zip")
)

# Crear carpetas necesarias
for p in (LOGS_DIR, JOBS_DIR):
    p.mkdir(parents=True, exist_ok=True)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _to_label_and_conf(probs_bhs: Dict[str, float]) -> Tuple[str, float]:
    """
    probs_bhs: {"buy":p,"hold":p,"sell":p} -> ("buy|hold|sell", confidence_en_%)
    """
    label = max(probs_bhs, key=probs_bhs.get)
    conf_pct = float(probs_bhs[label]) * 100.0
    return label, conf_pct


def _probs_to_table_keys(probs_bhs: Dict[str, float]) -> Dict[str, float]:
    """Mapea a claves usadas por el dashboard: '-1' (sell), '0' (hold), '1' (buy)."""
    return {
        "-1": float(probs_bhs.get("sell", 0.0)),
        "0": float(probs_bhs.get("hold", 0.0)),
        "1": float(probs_bhs.get("buy", 0.0)),
    }


def _r1m_from_prices(prices: List[float]) -> np.ndarray:
    p = np.asarray(prices, dtype=np.float64)
    if p.ndim != 1 or p.size < 3:
        return np.array([], dtype=np.float64)
    return np.diff(np.log(p))


def _agg_returns(r1m: np.ndarray, step: int) -> np.ndarray:
    """Suma rodante de retornos 1m para formar retornos de 'step' minutos (tipo 15m)."""
    if r1m.size < step:
        return np.array([], dtype=np.float64)
    kernel = np.ones(step, dtype=np.float64)
    return np.convolve(r1m, kernel, mode="valid")


def _zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    mu = np.mean(x) if x.size else 0.0
    sd = np.std(x) if x.size else 1.0
    if not np.isfinite(sd) or sd < 1e-8:
        sd = 1.0
    return (x - mu) / sd


def load_ohlcv_df(symbol: str, tf: str, connector_obj) -> pd.DataFrame:
    """
    Intenta cargar /app/data/ohlcv_{symbol}_{tf}.csv.
    Si tf=1m y no existe el CSV, usa el conector en vivo (fetch_ohlcv).
    Devuelve DataFrame con columnas: timestamp, open, high, low, close, volume (UTC).
    """
    symbol = symbol.replace("/", "").upper()
    tf = (tf or "1m").lower()
    if tf not in ALLOWED_TFS:
        raise ValueError(f"Timeframe no soportado: {tf}")

    csv_path = DATA_DIR / f"ohlcv_{symbol}_{tf}.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        needed = {"timestamp", "open", "high", "low", "close", "volume"}
        if not needed.issubset(df.columns):
            raise ValueError(f"CSV {csv_path.name} no tiene columnas OHLCV completas.")
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = (
            df.dropna(subset=["timestamp", "close"])
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
        if df.empty:
            raise ValueError(f"CSV {csv_path.name} está vacío.")
        return df

    # Fallback solo para 1m: en vivo
    if tf == "1m":
        ohlcv = connector_obj.fetch_ohlcv(symbol, timeframe="1m", limit=200)
        if not ohlcv:
            raise RuntimeError("fetch_ohlcv devolvió vacío.")
        df = pd.DataFrame(
            ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True, errors="coerce")
        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = (
            df.dropna(subset=["timestamp", "close"])
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
        if df.empty:
            raise RuntimeError("OHLCV en vivo sin filas válidas.")
        return df

    # Para 5m/15m/1h sin CSV:
    raise FileNotFoundError(f"No existe {csv_path.name}. Genera los CSVs para {tf}.")


# -----------------------
# Modelos de request
# -----------------------
class StockformerSignalRequest(BaseModel):
    symbol: str = Field("BTCUSDT")
    timeframes: List[str] = Field(default_factory=lambda: ["1m", "5m", "15m", "1h"])
    features: Optional[Dict[str, Any]] = None  # opcional
    horizon_min: int = Field(STOCKFORMER_HORIZON_MIN)


class MultimodalActionRequest(BaseModel):
    symbols: List[str]
    price_window: List[List[float]]  # lista por símbolo (usamos [0])
    sentiment: Optional[float] = 0.0


# -----------------------
# Endpoints básicos
# -----------------------
@app.get("/health")
def health():
    return {"status": "ok", "ts": _utc_now_iso()}


# -----------------------
# Señal: Stockformer
# -----------------------
# -----------------------
# Señal: Stockformer
# -----------------------
@app.post("/signal/stockformer")
def signal_stockformer(req: StockformerSignalRequest):
    """
    Devuelve probabilidades buy/hold/sell y meta (horizon=15m por defecto).
    - 1º intenta el modelo Stockformer plug-in.
    - Si falla, usa un fallback determinista con retornos 1m (sin AISignal).
    """
    try:
        symbol = (req.symbol or "BTCUSDT").replace("/", "").upper()
        tfs = [tf for tf in (req.timeframes or []) if tf in ALLOWED_TFS]
        if not tfs:
            tfs = ["1m", "5m", "15m", "1h"]

        source = "stockformer"
        # ======= Vía principal: modelo =======
        try:
            sf = get_stockformer()
            if req.features is not None:
                out = sf.predict_from_features(req.features)  # {'buy':p,'hold':p,'sell':p}
            else:
                dfs = {tf: load_ohlcv_df(symbol, tf, connector) for tf in tfs}
                out = sf.predict_from_ohlcv(dfs=dfs, horizon_min=req.horizon_min)
            probs_bhs = {
                "buy": float(out.get("buy", 0.0)),
                "hold": float(out.get("hold", 0.0)),
                "sell": float(out.get("sell", 0.0)),
            }
        except Exception:
            # ======= Fallback determinista (sin AISignal) =======
            source = "fallback_1m"
            df = load_ohlcv_df(symbol, "1m", connector)
            c = df["close"].to_numpy(dtype=np.float64)
            r = np.diff(np.log(c))
            if r.size < 64:
                # acolchamos para estabilidad
                pad = 64 - r.size
                if pad > 0:
                    r = np.concatenate([np.full(pad, r[0] if r.size else 0.0), r])
            mu = float(np.mean(r[-32:]))
            vol = float(np.std(r[-32:])) + 1e-8
            s = mu / vol  # señal normalizada
            # asignación suave tipo soft-sign
            bias = float(np.tanh(3.0 * s))  # [-1,1]
            p_buy = 0.33 + 0.33 * max(0.0, bias)
            p_sell = 0.33 + 0.33 * max(0.0, -bias)
            p_hold = 1.0 - (p_buy + p_sell)
            probs_bhs = {"buy": p_buy, "hold": max(1e-9, p_hold), "sell": p_sell}

        # Normalización defensiva
        ps = np.array([probs_bhs["sell"], probs_bhs["hold"], probs_bhs["buy"]], dtype=np.float64)
        if not np.isfinite(ps).all() or np.all(ps <= 0):
            ps = np.array([0.25, 0.5, 0.25], dtype=np.float64)
        ps = np.clip(ps, 1e-9, None)
        ps = ps / ps.sum()
        probs_bhs = {"sell": float(ps[0]), "hold": float(ps[1]), "buy": float(ps[2])}

        label, conf_pct = _to_label_and_conf(probs_bhs)
        meta = {
            "target_horizon_min": int(req.horizon_min),
            "timeframes": tfs,
            "ts": _utc_now_iso(),
            "model_path": STOCKFORMER_MODEL_FILE,
            "source": source,
        }
        return {
            "signal": label,
            "confidence_pct": conf_pct,
            "probs": _probs_to_table_keys(probs_bhs),  # "-1","0","1"
            "meta": meta,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



# -----------------------
# Agente RL multimodal
# -----------------------
_RL_MODEL_SB3 = None  # cache opcional para SB3


def _get_sb3_model():
    """
    Intenta cargar un modelo SB3 desde MULTIMODAL_RL_FILE.
    Devuelve (model, None) o (None, 'error string').
    """
    global _RL_MODEL_SB3
    if _RL_MODEL_SB3 is not None:
        return _RL_MODEL_SB3, None
    try:
        from stable_baselines3 import PPO  # o el algoritmo que se haya usado
        _RL_MODEL_SB3 = PPO.load(MULTIMODAL_RL_FILE, device="cpu")
        return _RL_MODEL_SB3, None
    except Exception as e:
        return None, str(e)


@app.post("/agent/multimodal/action")
def multimodal_action(req: MultimodalActionRequest):
    """
    Construye observación: últimos RL_LOOKBACK retornos agregados a RL_STEP_MIN + sentimiento.
    Devuelve acción actual, histórico discretizado y una equity curva de simulación corta.
    Fallback: modo "mock" si el modelo SB3 o de la clase MultimodalRLAgent no está disponible.
    """
    try:
        if not req.price_window or not req.price_window[0]:
            raise HTTPException(status_code=400, detail="price_window vacío")

        prices = req.price_window[0]
        r1m = _r1m_from_prices(prices)
        r_step = _agg_returns(r1m, RL_STEP_MIN)

        # Asegurar longitud mínima para formar la observación
        min_len = RL_LOOKBACK + 1
        if r_step.size < min_len:
            if r_step.size == 0:
                r_step = np.zeros(min_len, dtype=np.float64)
            else:
                pad = min_len - r_step.size
                r_step = np.concatenate([np.full(pad, r_step[0]), r_step])

        x = _zscore(r_step)[-RL_LOOKBACK:]  # (RL_LOOKBACK,)
        sentiment = float(req.sentiment or 0.0)
        obs = np.concatenate([x, [sentiment]], axis=0).reshape(1, -1)  # (1, RL_LOOKBACK+1)

        # 1) Intento con clase "plug-in" (si implementa predict)
        a_curr: Optional[float] = None
        agent_err: Optional[str] = None
        try:
            rl_agent = get_rl_agent()
            a_curr = float(np.clip(rl_agent.predict(obs), -1.0, 1.0))
        except Exception as e:
            agent_err = str(e)

        # 2) Intento con SB3 (PPO.load)
        mock = False
        if a_curr is None:
            model, err = _get_sb3_model()
            if model is not None:
                try:
                    pred = model.predict(obs, deterministic=True)[0]
                    a_curr = float(np.clip(pred, -1.0, 1.0))
                except Exception as e:
                    agent_err = f"SB3 predict: {e}"
            else:
                agent_err = err

        # 3) Fallback mock
        if a_curr is None:
            mock = True
            a_curr = float(np.tanh(obs.sum()))

        # Historial corto para UI
        K = int(min(128, r_step.size))
        actions_hist: List[float] = []

        if mock:
            actions_hist = [a_curr] * K
        else:
            # Ventanas deslizantes con normalización por ventana
            for i in range(-K, 0):
                end = r_step.size + i
                start = end - RL_LOOKBACK
                if start < 0:
                    continue
                x_i = _zscore(r_step[start:end])
                o_i = np.concatenate([x_i, [sentiment]], axis=0).reshape(1, -1)
                try:
                    if _RL_MODEL_SB3 is not None:
                        a_i = float(np.clip(_RL_MODEL_SB3.predict(o_i, deterministic=True)[0], -1.0, 1.0))
                    else:
                        # Intento con clase plug-in
                        rl_agent = get_rl_agent()
                        a_i = float(np.clip(rl_agent.predict(o_i), -1.0, 1.0))
                except Exception:
                    a_i = a_curr
                actions_hist.append(a_i)

            if not actions_hist:
                actions_hist = [a_curr] * K

        actions_hist = np.asarray(actions_hist, dtype=np.float64)
        positions_hist = np.where(
            actions_hist > RL_THETA, 1, np.where(actions_hist < -RL_THETA, -1, 0)
        ).astype(int).tolist()

        # Simulación de equity sobre los últimos K retornos agregados
        r_sim = r_step[-len(positions_hist):]
        eq = [1.0]
        for i, ret in enumerate(r_sim):
            pos = positions_hist[i]
            eq.append(eq[-1] * (1.0 + pos * float(ret)))
        equity_curve = eq  # longitud = len(positions_hist)+1

        meta = {
            "backend": "sb3" if not mock else "mock",
            "mock": mock,
            "theta": RL_THETA,
            "lookback": RL_LOOKBACK,
            "step_min": RL_STEP_MIN,
            "model_path": MULTIMODAL_RL_FILE,
            "obs_dim": int(obs.shape[1]),
            "scaling": "returns_step_norm",
            "action_stats": {
                "min": float(np.min(actions_hist)) if actions_hist.size else 0.0,
                "mean": float(np.mean(actions_hist)) if actions_hist.size else 0.0,
                "max": float(np.max(actions_hist)) if actions_hist.size else 0.0,
            },
            "errors": [agent_err] if (agent_err and mock) else [],
        }

        return {
            "symbols": req.symbols,
            "action": [a_curr],
            "positions_hist": positions_hist,
            "equity_curve": equity_curve,
            "meta": meta,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------
# Reload modelos
# -----------------------
@app.post("/models/reload")
def models_reload(target: str = Query("all", regex="^(stockformer|multimodal|all)$")):
    """
    Recarga pesos en caliente.
    """
    global _stockformer, _rl_agent, _RL_MODEL_SB3
    res = {"stockformer": False, "multimodal": False}

    if target in ("stockformer", "all"):
        try:
            _stockformer = None
            get_stockformer()  # fuerza load()
            res["stockformer"] = True
        except Exception:
            res["stockformer"] = False

    if target in ("multimodal", "all"):
        try:
            _rl_agent = None
            _RL_MODEL_SB3 = None
            get_rl_agent()  # fuerza load()
            # Carga SB3 si existe
            _get_sb3_model()
            res["multimodal"] = True
        except Exception:
            res["multimodal"] = False

    return {"reloaded": res, "ts": _utc_now_iso()}


# -----------------------
# Lanzar entrenamientos (background)
# -----------------------
def _is_pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def _save_job_record(rec: Dict[str, Any]) -> None:
    (JOBS_DIR / f"{rec['job_id']}.json").write_text(json.dumps(rec, ensure_ascii=False))


def _load_job_record(job_id: str) -> Optional[Dict[str, Any]]:
    p = JOBS_DIR / f"{job_id}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _list_jobs() -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for fp in JOBS_DIR.glob("*.json"):
        try:
            rec = json.loads(fp.read_text())
            rec["alive"] = _is_pid_alive(int(rec.get("pid", -1)))
            out.append(rec)
        except Exception:
            continue
    out.sort(key=lambda r: r.get("started_at", ""), reverse=True)
    return out


def _tail_log(path: Path, n: int = 200) -> List[str]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        return lines[-n:]
    except Exception:
        return []


def _launch_training(kind: str, script_rel_path: str, args: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Lanza un entrenamiento en segundo plano:
    - kind: 'stockformer' | 'multimodal'
    - script_rel_path: p.ej. 'src/ai_training/train_stockformer.py'
    - args: lista opcional de args (e.g. ["--symbol","BTCUSDT","--epochs","3"])
    Devuelve dict con job_id, pid, log_path.
    """
    args = args or []
    job_id = f"{kind}-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:6]}"
    log_path = LOGS_DIR / f"train_{job_id}.log"
    script_path = Path(script_rel_path)

    # Comando: python -u <script> <args...>
    cmd = ["python", "-u", str(script_path)] + list(args)
    with log_path.open("w", encoding="utf-8") as lf:
        proc = subprocess.Popen(cmd, stdout=lf, stderr=lf, cwd=str(Path.cwd()))
    rec = {
        "job_id": job_id,
        "kind": kind,
        "cmd": cmd,
        "pid": proc.pid,
        "started_at": _utc_now_iso(),
        "log_path": str(log_path),
    }
    _save_job_record(rec)
    return rec


class TrainRequest(BaseModel):
    args: Optional[List[str]] = None  # e.g., ["--symbol","BTCUSDT","--epochs","3"]


@app.post("/train/stockformer")
def train_stockformer(req: TrainRequest):
    try:
        rec = _launch_training(
            "stockformer",
            "src/ai_training/train_stockformer.py",
            args=req.args or [],
        )
        return {"ok": True, "job": rec}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/train/multimodal")
def train_multimodal(req: TrainRequest):
    try:
        rec = _launch_training(
            "multimodal",
            "src/ai_training/train_multimodal.py",
            args=req.args or [],
        )
        return {"ok": True, "job": rec}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/train/jobs")
def list_jobs():
    return {"jobs": _list_jobs()}


@app.get("/train/jobs/{job_id}")
def get_job(job_id: str):
    rec = _load_job_record(job_id)
    if not rec:
        raise HTTPException(status_code=404, detail="job_id no encontrado")
    rec["alive"] = _is_pid_alive(int(rec.get("pid", -1)))
    return rec


@app.get("/train/logs/{job_id}")
def get_job_logs(job_id: str, n: int = 200):
    rec = _load_job_record(job_id)
    if not rec:
        raise HTTPException(status_code=404, detail="job_id no encontrado")
    lines = _tail_log(Path(rec["log_path"]), n=n)
    return JSONResponse(
        content={"job_id": job_id, "lines": lines, "n": len(lines)},
        media_type="application/json",
    )


# -----------------------
# Ingesta / Sentimiento
# -----------------------
class IngestRequest(BaseModel):
    symbol: Optional[str] = None  # por si se parametriza a futuro


@app.post("/ingest/news")
def api_ingest_news(req: IngestRequest):
    """
    Lanza ingesta rápida de fuentes RSS/Atom definidas en data_pipeline.news_sources.
    """
    try:
        ingest_sources()  # guarda CSV base de noticias
        return {"ok": True, "ts": _utc_now_iso()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class BuildSentimentRequest(BaseModel):
    symbol: Optional[str] = None


@app.post("/build/sentiment")
def api_build_sentiment(req: BuildSentimentRequest):
    """
    Reconstruye la serie de sentimiento y los "drivers" (top enlaces) y los guarda:
    - SENTIMENT_CSV (serie diaria con decaimiento)
    - DRIVERS_CSV   (enlaces que más aportan por día)
    """
    try:
        symbol = (req.symbol or DEFAULT_SYMBOL).upper()
        out = build_sentiment(symbol=symbol, sentiment_csv=SENTIMENT_CSV, drivers_csv=DRIVERS_CSV)
        return {"ok": True, "symbol": symbol, "files": out, "ts": _utc_now_iso()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
