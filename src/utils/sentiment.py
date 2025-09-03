# src/utils/sentiment.py
from __future__ import annotations
import math
import pandas as pd
from typing import Iterable, Tuple, Optional

# FinBERT (opcional): pip install transformers torch --no-cache-dir
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
    _HAS_NLP = True
except Exception:
    _HAS_NLP = False

def load_finbert_pipeline():
    if not _HAS_NLP:
        return None
    tok = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    mdl = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    return TextClassificationPipeline(model=mdl, tokenizer=tok, return_all_scores=True, truncation=True)

def value_embedding(pos: float, neg: float, neu: float) -> float:
    # Evita divisiones por cero
    eps = 1e-9
    ratio = (pos + eps) / (neg + eps)
    v = math.tanh( (ratio) / max(neu, eps) )
    # Limita a [-1,1] y centra: más neg -> valor cerca de -1
    # Nota: FinBERT devuelve probabilidades por etiqueta; aquí ratio pos/neg ya codifica el signo.
    return max(-1.0, min(1.0, v))

def exponential_decay_forward_fill(s: pd.Series, gamma: float = 0.8) -> pd.Series:
    """
    s: serie indexada por fecha (diaria), con NaNs los días sin noticia.
    y_t = a * (1 - gamma)^t  ; gamma in (0,1)
    """
    s = s.copy()
    last_val = None
    last_idx = None
    out = []
    for idx, v in s.items():
        if pd.notna(v):
            last_val, last_idx = float(v), idx
            out.append(last_val)
        else:
            if last_val is None:
                out.append(None)
            else:
                dt = (idx - last_idx).days if last_idx is not None else 0
                out.append( last_val * ((1.0 - gamma) ** max(0, dt)) )
    return pd.Series(out, index=s.index, dtype="float64")

def headlines_to_daily_sentiment(df: pd.DataFrame, date_col="date", sym_col="symbol", text_col="headline",
                                 symbol="BTCUSDT", gamma: float = 0.8) -> pd.DataFrame:
    """
    df: columnas mínimas [date, symbol, headline]. 'date' en UTC (naive o tz-aware).
    Devuelve DataFrame con columnas [date, sentiment] para el símbolo.
    """
    if not _HAS_NLP:
        # fallback: todo neutro
        out = (df.assign(date=pd.to_datetime(df[date_col]).dt.date)
                 .query(f"{sym_col} == @symbol")
                 .groupby("date", as_index=False)
                 .agg(sentiment=("headline", lambda x: 0.0)))
        out["sentiment"] = exponential_decay_forward_fill(out.set_index("date")["sentiment"], gamma=gamma).values
        return out

    nlp = load_finbert_pipeline()
    df = df.copy()
    df["date"] = pd.to_datetime(df[date_col], utc=True).dt.tz_convert(None).dt.date
    df = df[df[sym_col] == symbol]
    if df.empty:
        return pd.DataFrame(columns=["date", "sentiment"])

    scores = []
    for txt in df[text_col].astype(str).tolist():
        res = nlp(txt)[0]  # lista de dicts: [{'label':'negative','score':...}, ...]
        d = {r["label"].lower(): float(r["score"]) for r in res}
        pos, neg, neu = d.get("positive", 0.0), d.get("negative", 0.0), d.get("neutral", 1e-9)
        scores.append(value_embedding(pos, neg, neu))
    df["ve"] = scores

    daily = df.groupby("date", as_index=False).agg(sentiment=("ve", "mean"))
    # reindex diario y aplicar decaimiento
    idx = pd.date_range(min(daily["date"]), max(daily["date"]), freq="D").date
    s = daily.set_index("date")["sentiment"].reindex(idx)
    s = exponential_decay_forward_fill(s, gamma=gamma)
    return pd.DataFrame({"date": s.index, "sentiment": s.values})
