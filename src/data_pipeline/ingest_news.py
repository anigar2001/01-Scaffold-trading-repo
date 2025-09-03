# src/data_pipeline/ingest_news.py
from __future__ import annotations
import os, re, unicodedata
from datetime import timezone
from typing import Dict, List
import feedparser, pandas as pd
from dateutil import parser as dtp

from .news_sources import SOURCES, DEFAULT_SYMBOL, HEADLINES_CSV

DATA_DIR = os.getenv("DATA_DIR", "/app/data")
OUT_PATH = os.path.join(DATA_DIR, HEADLINES_CSV)

def _normalize_text(s: str) -> str:
    s = (s or "").strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = re.sub(r"\s+", " ", s)
    return s

def _norm_row(dt_str: str, title: str, src_name: str, link: str, symbol: str) -> Dict:
    try:
        ts = dtp.parse(dt_str).astimezone(timezone.utc).replace(tzinfo=None)
        d = ts.date().isoformat()
    except Exception:
        d = pd.Timestamp.utcnow().date().isoformat()
    return {
        "date": d,
        "symbol": symbol,
        "headline": f"[{src_name}] {title}".strip(),
        "source": src_name,
        "url": link or "",
        "key": _normalize_text((title or "") + " " + (link or "")),
    }

def ingest_sources(symbol: str = DEFAULT_SYMBOL) -> Dict:
    rows: List[Dict] = []
    for s in SOURCES:
        feed = feedparser.parse(s["url"])
        src_name = s.get("name") or feed.feed.get("title", "rss")
        for e in feed.entries[:300]:
            rows.append(_norm_row(
                e.get("published") or e.get("updated") or "",
                e.get("title",""),
                src_name,
                e.get("link",""),
                symbol
            ))

    df = pd.DataFrame(rows).dropna(subset=["date","headline"])
    if os.path.exists(OUT_PATH):
        old = pd.read_csv(OUT_PATH)
        if "key" not in old.columns:
            old["key"] = (old["headline"].astype(str).str.lower().str.strip() + " " + old.get("url","").astype(str).str.lower())
        df = pd.concat([old, df], ignore_index=True)

    # Deduplicado conservador por 'key'
    df = df.drop_duplicates(subset=["key"], keep="first")
    df = df[["date","symbol","headline","source","url"]].sort_values(["date","source"]).reset_index(drop=True)

    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    return {"ok": True, "file": OUT_PATH, "rows": int(len(df))}
