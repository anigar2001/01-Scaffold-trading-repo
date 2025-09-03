# src/api.py

import os
import csv
import json
import uuid
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

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

# Construir sentimientos
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

DATA_DIR = Path(os.getenv("DATA_DIR", "/app/data"))
LOGS_DIR = Path(os.getenv("LOGS_DIR", "/app/logs"))
JOBS_DIR = DATA_DIR / "train_jobs"
for p in (LOGS_DIR, JOBS_DIR):
    p.mkdir(parents=True, exist_ok=True)


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
        df = df.dropna(subset=["timestamp", "close"]).sort_values("timestamp").reset_index(drop=True)
        if df.empty:
            raise ValueError(f"CSV {csv_path.name} está vacío.")
        return df

    # Fallback solo para 1m: en vivo
    if tf == "1m":
        ohlcv = connector_obj.fetch_ohlcv(symbol, timeframe="1m", limit=200)
        if not ohlcv:
            raise RuntimeError("fetch_ohlcv devolvió vacío.")
        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True, errors="coerce")
        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["timestamp", "close"]).sort_values("timestamp").reset_index(drop=True)
        if df.empty:
            raise RuntimeError("OHLCV en vivo sin filas válidas.")
        return df

    # Para 5m/15m/1h sin CSV:
    raise FileNotFoundError(f"No existe {csv_path.name}. Genera los CSVs para {tf}.")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


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
    # ordena por fecha desc
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
      - args: lista opcional de args
    Devuelve dict con job_id, pid, log_path.
    """
    args = args or []
    job_id = f"{kind}-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:6]}"
    log_path = LOGS_DIR / f"train_{job_id}.log"
    script_path = Path(script_rel_path)

    # Comando: python <script> [args...]
    # Nota: el script maneja falta de librerías guardando artefacto dummy.
    cmd = ["python", str(script_path)] + args

    # Abre log y lanza
    log_f = open(log_path, "ab", buffering=0)
    proc = subprocess.Popen(
        cmd,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        cwd=str(Path.cwd()),
        start_new_session=True,  # separa del proceso del servidor
        env=os.environ.copy(),
    )

    rec = {
        "job_id": job_id,
        "kind": kind,
        "pid": proc.pid,
        "script": str(script_path),
        "args": args,
        "log_path": str(log_path),
        "started_at": _utc_now_iso(),
    }
    _save_job_record(rec)
    return rec


# -----------------------
# Modelos Pydantic
# -----------------------
class OrderRequest(BaseModel):
    symbol: str
    side: str          # 'buy' o 'sell'
    type: str          # 'market' o 'limit'
    amount: float
    price: float | None = None  # opcional, solo para limit


# Stockformer (POST opcional, para futuros front-ends)
class StockformerFeatures(BaseModel):
    symbol: str = Field(..., description="Símbolo, p.ej. BTCUSDT")
    timeframes: Optional[List[str]] = Field(None, description="['1m','5m','15m','1h'] (opcional)")
    features: Dict[str, Any] = Field(default_factory=dict, description="Mapa TF->listas o dicts de features")


class StockformerResp(BaseModel):
    symbol: str
    probs: Dict[str, float]
    attention_by_tf: Dict[str, float]
    meta: Dict[str, Any]


# RL Multimodal (POST opcional)
class MultimodalState(BaseModel):
    symbols: Optional[List[str]] = None
    price_window: Optional[List[List[float]]] = None
    sentiment: Optional[Any] = None
    sec_sentiment: Optional[Any] = None


class RLActionResp(BaseModel):
    symbols: List[str]
    action: List[float]
    equity_curve: Optional[List[float]] = None
    positions_hist: Optional[List[int]] = None
    meta: Dict[str, Any]


# -----------------------
# Endpoints base
# -----------------------
@app.get("/", include_in_schema=False)
def root():
    return {"status": "ok", "service": "Trading Bot API", "docs": "/docs"}


@app.get("/health", include_in_schema=False)
def health():
    return {"ok": True}


@app.get("/markets")
def list_markets(q: str | None = Query(None, description="Filtro contiene (opcional)"),
                 limit: int = Query(200, ge=1, le=2000)):
    try:
        m = connector.exchange.load_markets()
        symbols = list(m.keys())
        if q:
            q_low = q.strip().upper()
            symbols = [s for s in symbols if q_low in s.upper()]
        return symbols[:limit]
    except Exception as e:
        return {"error": str(e)}


@app.get("/balance")
def get_balance():
    try:
        return connector.get_balance()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ticker/{symbol}")
def get_ticker(symbol: str):
    try:
        symbol = symbol.replace("/", "").upper()
        return connector.get_ticker(symbol)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/order")
def create_order(order: OrderRequest):
    try:
        return order_manager.create_order(
            symbol=order.symbol,
            side=order.side,
            type=order.type,
            amount=order.amount,
            price=order.price
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/trades")
def get_trades():
    trades = []
    try:
        with open(LOG_FILE, mode="r") as file:
            reader = csv.DictReader(file)
            for row in reader:
                trades.append(row)
    except FileNotFoundError:
        return JSONResponse(content={"message": "No hay operaciones registradas"}, status_code=200)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    return trades


# -----------------------
# Señal IA (compatible con tu dashboard actual)
# -----------------------
@app.get("/signal/{symbol}")
def get_signal(symbol: str, tf: Optional[str] = Query(default=None, description="1m|5m|15m|1h")):
    """
    - Sin tf: señal agregada (intenta usar AISignal; si falla, Stockformer con multi-TF).
    - Con tf: señal por timeframe usando Stockformer.
    Respuesta incluye: signal ('buy'|'hold'|'sell'), confidence (0..100),
    probs {'-1':p_sell,'0':p_hold,'1':p_buy}, y last_ts (string).
    """
    symbol = symbol.replace("/", "").upper()

    try:
        # Con TF → Stockformer por timeframe
        if tf:
            tf_low = tf.lower()
            if tf_low not in ALLOWED_TFS:
                return {"detail": f"tf inválido: {tf}"}
            sm = get_stockformer()
            payload = {"symbol": symbol, "timeframes": [tf_low], "features": {tf_low: []}}
            out = sm.predict(payload)
            label, conf_pct = _to_label_and_conf(out.probs)
            return {
                "symbol": symbol,
                "timeframe": tf_low,
                "signal": label,
                "confidence": round(conf_pct, 2),
                "probs": _probs_to_table_keys(out.probs),
                "last_ts": "-",
            }

        # Sin TF → primero intenta AISignal (compat), si falla, Stockformer agregado
        ai_obj, err = get_ai()
        if ai_obj is not None:
            res = ai_obj.predict_signal(symbol)
            if isinstance(res, dict) and "signal" in res and "confidence" in res:
                return res
            # Fallback si AISignal no cumple formato

        sm = get_stockformer()
        payload = {
            "symbol": symbol,
            "timeframes": ["1m", "5m", "15m", "1h"],
            "features": {"1m": [], "5m": [], "15m": [], "1h": []},
        }
        out = sm.predict(payload)
        label, conf_pct = _to_label_and_conf(out.probs)
        return {
            "symbol": symbol,
            "signal": label,
            "confidence": round(conf_pct, 2),
            "probs": _probs_to_table_keys(out.probs),
            "last_ts": "-",
        }

    except Exception as e:
        import traceback, sys
        traceback.print_exc(file=sys.stderr)
        return {"error": f"Error señal IA: {e}"}


# -----------------------
# Endpoints opcionales (para futuros front-ends)
# -----------------------
@app.post("/signal/stockformer", response_model=StockformerResp)
def post_signal_stockformer(payload: StockformerFeatures):
    sm = get_stockformer()
    out = sm.predict(payload.dict())
    return StockformerResp(
        symbol=out.symbol,
        probs=out.probs,
        attention_by_tf=out.attention_by_tf,
        meta=out.meta,
    )


@app.post("/agent/multimodal/action", response_model=RLActionResp)
def post_multimodal_action(state: MultimodalState):
    agent = get_rl_agent()
    out = agent.act(state.dict())
    return RLActionResp(
        symbols=out.symbols,
        action=out.action,
        equity_curve=out.equity_curve,
        positions_hist=out.positions_hist,
        meta=out.meta,
    )


# -----------------------
# NUEVO: Entrenamiento on-demand + gestión de jobs
# -----------------------
class TrainResp(BaseModel):
    job_id: str
    pid: int
    kind: str
    log_path: str
    started_at: str


@app.post("/train/stockformer", response_model=TrainResp)
def train_stockformer(symbol: str = Query(default="BTCUSDT", description="Símbolo (por ahora fijo)")):
    """
    Lanza entrenamiento de Stockformer en background.
    Artefacto destino: $STOCKFORMER_MODEL_FILE (o /app/data/stockformer_model.pt).
    """
    rec = _launch_training(kind="stockformer", script_rel_path="src/ai_training/train_stockformer.py")
    return TrainResp(
        job_id=rec["job_id"], pid=rec["pid"], kind=rec["kind"],
        log_path=rec["log_path"], started_at=rec["started_at"]
    )


@app.post("/train/multimodal", response_model=TrainResp)
def train_multimodal():
    """
    Lanza entrenamiento del agente RL multimodal en background.
    Artefacto destino: $MULTIMODAL_RL_FILE (o /app/data/multimodal_rl.zip).
    """
    rec = _launch_training(kind="multimodal", script_rel_path="src/ai_training/train_multimodal.py")
    return TrainResp(
        job_id=rec["job_id"], pid=rec["pid"], kind=rec["kind"],
        log_path=rec["log_path"], started_at=rec["started_at"]
    )


@app.get("/train/status")
def train_status():
    """
    Lista los jobs lanzados y su estado (alive=True si el PID sigue activo).
    """
    return _list_jobs()


@app.get("/train/log/{job_id}")
def train_log(job_id: str, tail: int = Query(default=200, ge=1, le=5000)):
    """
    Devuelve el tail (últimas N líneas) del log del job.
    """
    rec = _load_job_record(job_id)
    if not rec:
        raise HTTPException(status_code=404, detail="job_id no encontrado")
    lines = _tail_log(Path(rec["log_path"]), n=tail)
    return {"job_id": job_id, "lines": lines}


@app.post("/models/reload")
def models_reload(target: str = Query(default="all", regex="^(stockformer|multimodal|all)$")):
    """
    Recarga pesos en caliente sin reiniciar contenedor.
    - target=stockformer: recarga Stockformer
    - target=multimodal:  recarga RL
    - target=all:         ambos
    """
    reloaded: Dict[str, bool] = {}
    if target in ("stockformer", "all"):
        sm = get_stockformer()
        reloaded["stockformer"] = sm.load()
    if target in ("multimodal", "all"):
        rl = get_rl_agent()
        reloaded["multimodal"] = rl.load()
    return {"reloaded": reloaded}

@app.post("/data/news/ingest")
def data_news_ingest(symbol: str = Query(default=DEFAULT_SYMBOL, description="Símbolo (BTCUSDT)")):
    res = ingest_sources(symbol=symbol)
    return res

@app.post("/data/sentiment/build")
def data_sentiment_build(symbol: str = Query(default=DEFAULT_SYMBOL, description="Símbolo (BTCUSDT)")):
    res = build_sentiment(symbol=symbol)
    return res

@app.get("/data/sentiment/{symbol}")
def data_sentiment_get(symbol: str, last_n: int = Query(default=200, ge=1, le=5000)):
    import os, pandas as pd
    DATA_DIR = os.getenv("DATA_DIR", "/app/data")
    path = os.path.join(DATA_DIR, SENTIMENT_CSV)
    if not os.path.exists(path):
        return {"error": f"No existe {path}. Ejecuta /data/news/ingest y /data/sentiment/build."}
    df = pd.read_csv(path)
    if df.empty:
        return {"error": "Serie de sentimiento vacía."}
    df = df.tail(last_n)
    return {"symbol": symbol.upper(), "rows": df.to_dict(orient="records")}

@app.get("/data/sentiment/drivers/{symbol}")
def data_sentiment_drivers_get(symbol: str, last_n: int = Query(default=30, ge=1, le=500)):
    import os, pandas as pd
    DATA_DIR = os.getenv("DATA_DIR", "/app/data")
    path = os.path.join(DATA_DIR, DRIVERS_CSV)
    if not os.path.exists(path):
        return {"error": f"No existe {path}. Ejecuta /data/sentiment/build."}
    df = pd.read_csv(path)
    if df.empty:
        return {"error": "Drivers vacío."}
    df = df.sort_values(["date","direction"], ascending=[False, True]).head(last_n)
    return {"symbol": symbol.upper(), "rows": df.to_dict(orient="records")}

