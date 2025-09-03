# src/ai_training/build_sentiment.py
from __future__ import annotations
import os, math
import pandas as pd
from typing import Dict, List

from data_pipeline.news_sources import SOURCES, DEFAULT_SYMBOL, HEADLINES_CSV, SENTIMENT_CSV, DRIVERS_CSV

DATA_DIR = os.getenv("DATA_DIR", "/app/data")
HEADLINES_PATH = os.path.join(DATA_DIR, HEADLINES_CSV)
OUT_PATH = os.path.join(DATA_DIR, SENTIMENT_CSV)
OUT_DRIVERS = os.path.join(DATA_DIR, DRIVERS_CSV)

# FinBERT opcional
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
    _HAS_NLP = True
except Exception:
    _HAS_NLP = False

def _load_finbert():
    tok = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    mdl = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    return TextClassificationPipeline(model=mdl, tokenizer=tok, return_all_scores=True, truncation=True)

def _value_embedding(pos: float, neg: float, neu: float) -> float:
    eps = 1e-9
    v = math.tanh(((pos + eps) / (neg + eps)) / max(neu, eps))
    return max(-1.0, min(1.0, v))

def _exp_decay_forward_fill(s: pd.Series, gamma: float) -> pd.Series:
    s = s.copy()
    last_val, last_idx = None, None
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
                out.append(last_val * ((1.0 - gamma) ** max(0, dt)))
    return pd.Series(out, index=s.index, dtype="float64")

def build_sentiment(symbol: str = DEFAULT_SYMBOL) -> Dict:
    if not os.path.exists(HEADLINES_PATH):
        return {"ok": False, "error": f"No existe {HEADLINES_PATH}. Ejecuta la ingesta primero."}

    df = pd.read_csv(HEADLINES_PATH)
    if df.empty:
        return {"ok": False, "error": "Headlines vacío."}

    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df[df["symbol"] == symbol]

    nlp = _load_finbert() if _HAS_NLP else None

    # Puntuar cada titular (guardando headline y url)
    scored_rows: List[Dict] = []
    for r in df.itertuples(index=False):
        title = str(getattr(r, "headline", ""))
        source = str(getattr(r, "source", "")).lower()
        url = str(getattr(r, "url", ""))
        if nlp:
            res = nlp(title)[0]
            d = {x["label"].lower(): float(x["score"]) for x in res}
            ve = _value_embedding(d.get("positive",0.0), d.get("negative",0.0), d.get("neutral",1e-9))
        else:
            ve = 0.0
        scored_rows.append({"date": getattr(r, "date"), "source": source, "headline": title, "url": url, "ve": ve})

    scored = pd.DataFrame(scored_rows)
    if scored.empty:
        return {"ok": False, "error": "Sin VE calculables."}

    # Pesos/gamma por fuente
    w_map = {s["name"].lower(): float(s["weight"]) for s in SOURCES}
    g_map = {s["name"].lower(): float(s["gamma"]) for s in SOURCES}

    # Series diarias por fuente con decaimiento
    daily_src = scored.groupby(["date","source"], as_index=False).agg(sent=("ve","mean"))
    idx = pd.date_range(daily_src["date"].min(), daily_src["date"].max(), freq="D").date

    parts = []
    for name, gamma in g_map.items():
        w = w_map.get(name, 0.0)
        sub = daily_src[daily_src["source"].str.lower() == name][["date","sent"]].set_index("date").reindex(idx)
        dec = _exp_decay_forward_fill(sub["sent"], gamma=gamma)
        parts.append(pd.DataFrame({"date": idx, "source": name, "w": w, "sent": dec.values}))
    blended = pd.concat(parts, ignore_index=True)

    def _mix_day(g: pd.DataFrame) -> float:
        g2 = g.dropna(subset=["sent"])
        if g2.empty:
            return 0.0
        wsum = g2["w"].sum()
        return float((g2["sent"] * g2["w"]).sum() / (wsum if wsum > 0 else 1.0))

    out = blended.groupby("date", as_index=False).apply(_mix_day).rename(columns={None:"sentiment"})
    out = out[["date","sentiment"]]
    os.makedirs(DATA_DIR, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)

    # ---------------------------
    # DRIVERS: top contribuyente + y - por día (ve ponderado por peso)
    # ---------------------------
    # Unimos pesos por fuente y calculamos ve_w por titular
    scored["source"] = scored["source"].str.lower()
    scored["w"] = scored["source"].map(w_map).fillna(0.0)
    scored["ve_w"] = scored["ve"] * scored["w"]

    # Por día: top positivo y top negativo
    drivers_list = []
    for d, g in scored.groupby("date"):
        g2 = g.dropna(subset=["ve_w"])
        if g2.empty:
            continue
        pos = g2.loc[g2["ve_w"].idxmax()]
        neg = g2.loc[g2["ve_w"].idxmin()]
        # Solo guarda si aportan algo (peso>0)
        if float(pos["w"]) > 0:
            drivers_list.append({"date": d, "direction": "pos", "ve": float(pos["ve"]), "w": float(pos["w"]),
                                 "ve_w": float(pos["ve_w"]), "source": str(pos["source"]),
                                 "headline": str(pos["headline"]), "url": str(pos["url"])})
        if float(neg["w"]) > 0:
            drivers_list.append({"date": d, "direction": "neg", "ve": float(neg["ve"]), "w": float(neg["w"]),
                                 "ve_w": float(neg["ve_w"]), "source": str(neg["source"]),
                                 "headline": str(neg["headline"]), "url": str(neg["url"])})

    drv = pd.DataFrame(drivers_list)
    drv.to_csv(OUT_DRIVERS, index=False)

    return {"ok": True, "file": OUT_PATH, "drivers_file": OUT_DRIVERS, "rows": int(len(out)), "drivers_rows": int(len(drv))}
