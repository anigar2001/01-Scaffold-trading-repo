# src/dashboard/streamlit_app.py

import os
import time
import math
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import streamlit as st
import streamlit.components.v1 as components
from zoneinfo import ZoneInfo

# --- Compatibilidad autorefresh + auto-rerun por temporizador ---
# Si la versión de Streamlit no tiene st.autorefresh, crea un no-op
if not hasattr(st, "autorefresh"):
    try:
        st.autorefresh = lambda *args, **kwargs: None  # type: ignore[attr-defined]
    except Exception:
        pass

# Auto-refresh básico antes de construir la UI (usa _refresh_sec si existe)
try:
    _auto_interval = int(st.session_state.get("_refresh_sec", int(os.getenv("DEFAULT_REFRESH_SEC", "30"))))
    _now0 = time.time()
    _last0 = st.session_state.get("_last_refresh_ts")
    if _last0 is None:
        st.session_state["_last_refresh_ts"] = _now0
    elif _now0 - float(_last0) >= float(_auto_interval):
        st.session_state["_last_refresh_ts"] = _now0
        try:
            st.rerun()
        except Exception:
            st.experimental_rerun()
except Exception:
    pass
from datetime import datetime

# =======================
# Config
# =======================
API_BASE = os.getenv("API_BASE", "http://app:8000")
DATA_DIR = os.getenv("DATA_DIR", "/app/data")
SYMBOL   = os.getenv("DASH_SYMBOL", "BTCUSDT")
INITIAL_USDT = float(os.getenv("INITIAL_USDT", "0"))
INITIAL_BTC  = float(os.getenv("INITIAL_BTC", "0"))
TZ_LOCAL = ZoneInfo("Europe/Madrid")

st.set_page_config(page_title="📊 Trading Control", layout="wide")
st.title("📊 Trading Control Dashboard")

# =======================
# Sidebar
# =======================
with st.sidebar:
    st.header("⚙️ Panel de control")
    refresh_sec = st.number_input("Auto-refresh (segundos)", min_value=5, max_value=300, value=30, step=5)
    try:
        st.session_state["_refresh_sec"] = int(refresh_sec)
    except Exception:
        pass

    lookback = st.selectbox("Ventana gráfica precio", ["200 velas", "500 velas", "1500 velas", "Todo"], index=1)
    lb_map = {"200 velas": 200, "500 velas": 500, "1500 velas": 1500, "Todo": None}
    lookback_n = lb_map[lookback]

    auto_scale = st.checkbox("Autoescalar ejes en gráficas", value=True)
    show_ema = st.checkbox("Mostrar EMAs (20/50)", value=True)
    show_trade_labels = st.checkbox("Mostrar etiquetas BUY/SELL", value=False)

    alert_conf_thresh = st.slider("Umbral alerta IA (%)", min_value=50, max_value=95, value=70, step=1)
    dist_n = st.slider("N últimos trades para distribución", min_value=10, max_value=1000, value=200, step=10)

    st.caption(f"API_BASE: {API_BASE}")
    st.caption(f"DATA_DIR: {DATA_DIR}")
    st.caption(f"Símbolo: {SYMBOL}")
    if INITIAL_USDT or INITIAL_BTC:
        st.caption(f"Inicial: {INITIAL_USDT:,.2f} USDT + {INITIAL_BTC:,.6f} BTC")
    try:
        st.session_state["_refresh_sec"] = int(refresh_sec)
    except Exception:
        pass

# Disparador de auto-refresh del lado cliente (sin dependencias externas)
try:
    _interval_ms = int(refresh_sec) * 1000
    components.html(
        f"""
        <script>
        setTimeout(function() {{
            if (window && window.location) {{ window.location.reload(); }}
        }}, {_interval_ms});
        </script>
        """,
        height=0,
    )
except Exception:
    pass

# 🔁 Auto-refresh oficial de Streamlit
st.autorefresh(interval=refresh_sec * 1000, key="auto_refresh")

# =======================
# Helpers
# =======================
def safe_post_json(url: str, payload: dict | None = None, timeout: float = 20.0):
    try:
        r = requests.post(url, json=payload or {}, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def safe_request_json(url: str, timeout: float = 8.0):
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}

def read_ohlcv_csv(symbol: str, tf: str) -> pd.DataFrame | None:
    path = os.path.join(DATA_DIR, f"ohlcv_{symbol}_{tf}.csv")
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, parse_dates=["timestamp"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce") \
                            .dt.tz_convert(TZ_LOCAL).dt.tz_localize(None)
        for c in ["open","high","low","close","volume"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["timestamp","close"]).sort_values("timestamp")
        return df
    except Exception as e:
        st.error(f"Error leyendo {path}: {e}")
        return None

def price_y_range(df: pd.DataFrame, pad_ratio: float = 0.01):
    ymin = float(df["low"].min()) if "low" in df.columns else float(df["close"].min())
    ymax = float(df["high"].max()) if "high" in df.columns else float(df["close"].max())
    if math.isfinite(ymin) and math.isfinite(ymax) and ymax > ymin:
        pad = (ymax - ymin) * pad_ratio
        return [ymin - pad, ymax + pad]
    return None

def load_trades_df() -> pd.DataFrame | None:
    path = os.path.join(DATA_DIR, "trades.csv")
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        ts_col = next((k for k in ["timestamp","time","date"] if k in df.columns), None)
        if not ts_col:
            return None
        df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce") \
                        .dt.tz_convert(TZ_LOCAL).dt.tz_localize(None)
        df = df.rename(columns={ts_col:"timestamp"})
        for c in ["price","amount"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        if "side" in df.columns:
            df["side"] = df["side"].astype(str).str.lower().str.strip()
        if "symbol" in df.columns:
            df["symbol"] = df["symbol"].astype(str).str.replace("/","", regex=False).str.upper()
        df = df.dropna(subset=["timestamp","price","amount"]).sort_values("timestamp")
        return df
    except Exception as e:
        st.error(f"Error leyendo trades.csv: {e}")
        return None

def build_equity_curve(df_1m: pd.DataFrame, trades: pd.DataFrame,
                       init_usdt: float, init_btc: float) -> pd.DataFrame | None:
    if df_1m is None or trades is None or df_1m.empty or trades.empty:
        return None
    price = df_1m[["timestamp","close"]].rename(columns={"close":"price"}).copy()
    t = trades.copy()
    if "symbol" in t.columns:
        t = t[t["symbol"] == SYMBOL]
    if t.empty:
        return None
    t["btc_delta"]  = t["amount"].where(t["side"]=="buy", -t["amount"])
    t["usdt_delta"] = (-t["price"]*t["amount"]).where(t["side"]=="buy", t["price"]*t["amount"])
    t_agg = t.copy()
    t_agg["timestamp"] = t_agg["timestamp"].dt.floor("min")
    t_agg = t_agg.groupby("timestamp", as_index=False)[["btc_delta","usdt_delta"]].sum()
    tl = price.merge(t_agg, on="timestamp", how="left").fillna({"btc_delta":0.0,"usdt_delta":0.0})
    tl["btc_pos"]   = init_btc + tl["btc_delta"].cumsum()
    tl["usdt_cash"] = init_usdt + tl["usdt_delta"].cumsum()
    tl["equity_usdt"] = tl["usdt_cash"] + tl["btc_pos"] * tl["price"]
    tl["equity_btc"]  = tl["btc_pos"] + (tl["usdt_cash"] / tl["price"])
    return tl[["timestamp","price","btc_pos","usdt_cash","equity_usdt","equity_btc"]]

def compute_drawdown(equity: pd.Series) -> pd.DataFrame:
    roll_max = equity.cummax()
    dd = (equity - roll_max) / roll_max
    return pd.DataFrame({"drawdown": dd})

def simple_metrics_from_equity(eq_df: pd.DataFrame) -> dict:
    if eq_df is None or eq_df.empty:
        return {}
    eq = eq_df["equity_usdt"].astype(float)
    ret_total = (eq.iloc[-1]/eq.iloc[0] - 1.0) if eq.iloc[0] != 0 else 0.0
    dd_df = compute_drawdown(eq)
    max_dd = dd_df["drawdown"].min()
    rets = eq.pct_change().dropna()
    vol = rets.std() * math.sqrt(60*24*365) if not rets.empty else 0.0
    avg_ret = rets.mean() * (60*24*365) if not rets.empty else 0.0
    sharpe = (avg_ret / vol) if vol != 0 else 0.0
    return {"ret_total": ret_total, "max_dd": max_dd, "vol": vol, "sharpe": sharpe}

def enrich_trades_fifo(trades: pd.DataFrame) -> pd.DataFrame:
    if trades is None or trades.empty:
        return trades
    t = trades.copy()
    if "symbol" in t.columns:
        t = t[t["symbol"] == SYMBOL]
    buys = []
    rows = []
    for r in t.itertuples(index=False):
        side = getattr(r, "side", "")
        ts   = getattr(r, "timestamp")
        px   = float(getattr(r, "price", 0))
        amt  = float(getattr(r, "amount", 0))
        base = r._asdict()
        if side == "buy":
            buys.append({"timestamp": ts, "price": px, "amount": amt})
            base.update({"pnl_usdt": None, "duracion_min": None})
            rows.append(base)
        elif side == "sell":
            remaining = amt
            pnl_total = 0.0
            open_ts = None
            while remaining > 1e-12 and buys:
                b = buys[0]
                take = min(remaining, b["amount"])
                pnl_total += (px - b["price"]) * take
                open_ts = open_ts or b["timestamp"]
                b["amount"] -= take
                remaining -= take
                if b["amount"] <= 1e-12:
                    buys.pop(0)
            dur_min = None
            if open_ts is not None:
                try:
                    dur_min = (ts - open_ts).total_seconds()/60.0
                except Exception:
                    dur_min = None
            base.update({"pnl_usdt": pnl_total if amt>0 else None, "duracion_min": dur_min})
            rows.append(base)
        else:
            base.update({"pnl_usdt": None, "duracion_min": None})
            rows.append(base)
    return pd.DataFrame(rows)

def human_mtime(path: str) -> str | None:
    try:
        ts = os.path.getmtime(path)
        return datetime.fromtimestamp(ts, tz=TZ_LOCAL).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return None

def fmt_ts_ms(ts_ms: int | float | None) -> str:
    if ts_ms is None:
        return "-"
    try:
        return datetime.fromtimestamp(int(ts_ms)/1000, tz=TZ_LOCAL).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return str(ts_ms)

# =======================
# Carga base
# =======================
df_1m = read_ohlcv_csv(SYMBOL, "1m")
trades_df = load_trades_df()
equity_df = build_equity_curve(df_1m, trades_df, INITIAL_USDT, INITIAL_BTC) \
            if (df_1m is not None and trades_df is not None) else None

# ==========================================================
# (0) 💹 P&L (USDT) — antes del ticker + mini-panel de cartera
# ==========================================================
st.subheader("💹 P&L (USDT)")
if equity_df is not None and not equity_df.empty:
    start_val = float(equity_df["equity_usdt"].iloc[0])
    curr_val  = float(equity_df["equity_usdt"].iloc[-1])
    pnl_abs   = curr_val - start_val
    pnl_pct   = (curr_val/start_val - 1.0) * 100.0 if start_val != 0 else 0.0

    c1, c2, c3 = st.columns(3)
    c1.metric("Equity actual (USDT)", f"{curr_val:,.2f}")
    c2.metric("P&L", f"{pnl_abs:,.2f}", f"{pnl_pct:.2f}%")
    c3.metric("Equity inicial (USDT)", f"{start_val:,.2f}")

    # Mini-panel de cartera
    last = equity_df.iloc[-1]
    btc_pos   = float(last["btc_pos"])
    usdt_cash = float(last["usdt_cash"])
    last_px   = float(last["price"])
    equity_btc = float(last["equity_btc"])

    cA, cB, cC, cD = st.columns(4)
    cA.metric("BTC posición", f"{btc_pos:,.6f} BTC")
    cB.metric("USDT cash", f"{usdt_cash:,.2f} USDT")
    cC.metric("Last price", f"{last_px:,.2f} USDT")
    cD.metric("Equity (BTC)", f"{equity_btc:,.6f} BTC")
else:
    st.info("Sin datos de equity aún (necesita 1m + trades + iniciales).")

# ==========================================================
# (1) 🎯 Ticker
# ==========================================================
st.subheader("🎯 Ticker")
tk = safe_request_json(f"{API_BASE}/ticker/{SYMBOL}")
ticker_ts_local = None
if isinstance(tk, dict) and "error" not in tk and "detail" not in tk:
    last = tk.get("last") or tk.get("close") or tk.get("bid") or tk.get("ask")
    bid  = tk.get("bid")
    ask  = tk.get("ask")
    ts_h = fmt_ts_ms(tk.get("timestamp"))
    if tk.get("timestamp") is not None:
        try:
            ticker_ts_local = datetime.fromtimestamp(int(tk["timestamp"])/1000, tz=TZ_LOCAL).replace(tzinfo=None)
        except Exception:
            ticker_ts_local = None

    spread = spread_bps = None
    if bid is not None and ask is not None:
        try:
            spread = ask - bid
            spread_bps = (spread / ((ask+bid)/2)) * 10_000
        except Exception:
            pass

    c1,c2,c3,c4 = st.columns([1,1,1,1])
    c1.metric("Last", f"{last}")
    c2.metric("Bid", f"{bid}")
    c3.metric("Ask", f"{ask}")
    if spread is not None:
        c4.metric("Spread", f"{spread:.2f}" + (f" ({spread_bps:.2f} bps)" if spread_bps is not None else ""))
    st.caption(f"{tk.get('symbol','')} | ts: {ts_h} | baseVol: {tk.get('baseVolume','-')} | quoteVol: {tk.get('quoteVolume','-')}")
else:
    st.warning(f"Ticker no disponible: {tk.get('error', tk.get('detail','desconocido')) if isinstance(tk, dict) else tk}")

# ===================================
# (2) 🚨 Alertas de señal IA
# ===================================
st.subheader("🚨 Alertas de señal IA")
sig_now = safe_request_json(f"{API_BASE}/signal/{SYMBOL}")
if isinstance(sig_now, dict) and "signal" in sig_now:
    sig_lbl = sig_now.get("signal","-")
    conf = float(sig_now.get("confidence", 0) or 0)
    if sig_lbl in ("buy","sell") and conf >= alert_conf_thresh:
        (st.success if sig_lbl=="buy" else st.error)(
            f"ALERTA: {sig_lbl.upper()} con confianza {conf}% (umbral {alert_conf_thresh}%)"
        )
    else:
        st.info(f"Señal actual: {sig_lbl} ({conf}%). Umbral: {alert_conf_thresh}%")
else:
    st.warning(f"No se pudo obtener la señal IA: {sig_now.get('error','?') if isinstance(sig_now, dict) else sig_now}")

# ===========================================================================
# (3) 📈 Precio 1m (Madrid) + señales ejecutadas + overlay del ticker + EMAs + Volumen
# ===========================================================================
st.subheader(f"📈 Precio {SYMBOL} 1m (Madrid) + señales ejecutadas")
if df_1m is not None and not df_1m.empty:
    plot_df = df_1m.copy()
    if lookback_n is not None and len(plot_df) > lookback_n:
        plot_df = plot_df.tail(lookback_n)

    # EMAs
    if show_ema:
        plot_df["ema20"] = plot_df["close"].ewm(span=20, adjust=False).mean()
        plot_df["ema50"] = plot_df["close"].ewm(span=50, adjust=False).mean()

    # Base figure: velas (candlestick)
    fig_price = go.Figure()
    if set(["open","high","low","close"]).issubset(plot_df.columns):
        fig_price.add_trace(go.Candlestick(
            x=plot_df["timestamp"],
            open=plot_df["open"], high=plot_df["high"],
            low=plot_df["low"], close=plot_df["close"],
            name="Velas", yaxis="y1",
            increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
            showlegend=True,
        ))
    else:
        fig_price.add_trace(go.Scatter(x=plot_df["timestamp"], y=plot_df["close"],
                                       mode="lines", name="Close", yaxis="y1"))
    if show_ema:
        fig_price.add_trace(go.Scatter(x=plot_df["timestamp"], y=plot_df["ema20"], mode="lines", name="EMA20", yaxis="y1"))
        fig_price.add_trace(go.Scatter(x=plot_df["timestamp"], y=plot_df["ema50"], mode="lines", name="EMA50", yaxis="y1"))

    # Volumen (eje secundario)
    if "volume" in plot_df.columns:
        fig_price.add_trace(go.Bar(x=plot_df["timestamp"], y=plot_df["volume"],
                                   name="Volume", opacity=0.3, yaxis="y2"))

    # Trades
    if trades_df is not None and not trades_df.empty:
        t = trades_df.copy()
        if "symbol" in t.columns:
            t = t[t["symbol"] == SYMBOL]
        buys = t[t["side"]=="buy"]
        sells = t[t["side"]=="sell"]
        if not buys.empty:
            fig_price.add_trace(go.Scatter(x=buys["timestamp"], y=buys["price"],
                                           mode="markers", name="BUY",
                                           marker_symbol="triangle-up", marker_size=10, yaxis="y1"))
        if not sells.empty:
            fig_price.add_trace(go.Scatter(x=sells["timestamp"], y=sells["price"],
                                           mode="markers", name="SELL",
                                           marker_symbol="triangle-down", marker_size=10, yaxis="y1"))
        if show_trade_labels:
            for r in buys.itertuples(index=False):
                fig_price.add_annotation(x=r.timestamp, y=r.price, text="BUY", showarrow=True, arrowhead=2, ay=-25)
            for r in sells.itertuples(index=False):
                fig_price.add_annotation(x=r.timestamp, y=r.price, text="SELL", showarrow=True, arrowhead=2, ay=25)

    # Overlay del ticker (línea vertical)
    if tk and isinstance(tk, dict):
        ts_ms = tk.get("timestamp")
        if ts_ms:
            try:
                ticker_ts_local = datetime.fromtimestamp(int(ts_ms)/1000, tz=TZ_LOCAL).replace(tzinfo=None)
                tmin, tmax = plot_df["timestamp"].min(), plot_df["timestamp"].max()
                if tmin <= ticker_ts_local <= tmax:
                    fig_price.add_shape(type="line", x0=ticker_ts_local, x1=ticker_ts_local,
                                        y0=0, y1=1, yref="paper",
                                        line=dict(width=1.5, dash="dot"))
                    fig_price.add_annotation(x=ticker_ts_local, y=plot_df["close"].iloc[-1],
                                             text="Último tick", showarrow=True, arrowhead=2, ax=20, ay=-30)
            except Exception:
                pass

    # Layout (autorange vs rango fijo)
    layout_kwargs = dict(
        margin=dict(l=10, r=10, t=35, b=10),
        height=460,
        legend=dict(orientation="h"),
        yaxis=dict(title="Price", autorange=True),
        yaxis2=dict(title="Volume", overlaying="y", side="right", showgrid=False, autorange=True),
        xaxis=dict(autorange=True),
    )
    if not auto_scale:
        yr = price_y_range(plot_df)
        if yr:
            layout_kwargs["yaxis"].update(range=yr)  # type: ignore

    fig_price.update_layout(**layout_kwargs)
    st.plotly_chart(fig_price, use_container_width=True)
else:
    st.warning("No hay datos 1m todavía.")

# ============================================
# (4) 🧭 Señal IA por timeframe (ahora mismo) — con confianza por clase
# ============================================
st.subheader("🧭 Señal IA por timeframe (ahora mismo)")
rows = []
for tf in ["1m","5m","15m","1h"]:
    si = safe_request_json(f"{API_BASE}/signal/{SYMBOL}?tf={tf}")
    if isinstance(si, dict) and "signal" in si:
        probs = si.get("probs", {}) or {}
        sell_p = float(probs.get("-1", 0.0)) * 100.0
        hold_p = float(probs.get("0", 0.0)) * 100.0
        buy_p  = float(probs.get("1", 0.0)) * 100.0
        rows.append({
            "timeframe": tf,
            "signal": si.get("signal","-"),
            "conf (elegida) %": si.get("confidence", 0),
            "buy %": round(buy_p, 2),
            "hold %": round(hold_p, 2),
            "sell %": round(sell_p, 2),
            "last_ts": si.get("last_ts","-")
        })
    else:
        rows.append({"timeframe": tf, "signal": "N/A", "conf (elegida) %": None,
                     "buy %": None, "hold %": None, "sell %": None, "last_ts": "-"})
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ==================================
# (5) 💰 Equity (USDT) y Riesgo
# ==================================
st.subheader("💰 Equity (USDT) y Riesgo")
if equity_df is not None and not equity_df.empty:
    fig_eq = px.line(equity_df, x="timestamp", y="equity_usdt", title="Equity (USDT)")
    fig_eq.update_layout(
        margin=dict(l=10, r=10, t=35, b=10), height=300,
        xaxis=dict(autorange=True), yaxis=dict(autorange=True) if auto_scale else {}
    )
    st.plotly_chart(fig_eq, use_container_width=True)

    dd_df = compute_drawdown(equity_df["equity_usdt"])
    dd_df = dd_df.join(equity_df["timestamp"]).set_index("timestamp").reset_index()
    fig_dd = px.area(dd_df, x="timestamp", y="drawdown", title="Drawdown (proporción)")
    fig_dd.update_layout(
        margin=dict(l=10, r=10, t=35, b=10), height=220,
        xaxis=dict(autorange=True), yaxis=dict(autorange=True)
    )
    st.plotly_chart(fig_dd, use_container_width=True)

    mets = simple_metrics_from_equity(equity_df)
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Retorno total", f"{mets.get('ret_total',0)*100:.2f}%")
    c2.metric("Max drawdown", f"{mets.get('max_dd',0)*100:.2f}%")
    c3.metric("Vol anualizada", f"{mets.get('vol',0)*100:.2f}%")
    c4.metric("Sharpe (aprox)", f"{mets.get('sharpe',0):.2f}")

    # Semáforo de riesgo (simple): combina drawdown y volatilidad
    try:
        dd = abs(float(mets.get('max_dd',0)))
        vol = abs(float(mets.get('vol',0)))
        # Umbrales orientativos: ajusta a tu gusto
        if dd < 0.1 and vol < 0.5:
            color, label = "#2e7d32", "Riesgo BAJO"
        elif dd < 0.2 and vol < 1.0:
            color, label = "#f9a825", "Riesgo MEDIO"
        else:
            color, label = "#c62828", "Riesgo ALTO"
        st.markdown(f"<div style='padding:8px;border-radius:6px;background:{color};color:white;width:200px;text-align:center'>"+label+"</div>", unsafe_allow_html=True)
    except Exception:
        pass
else:
    st.info("Equity aún no disponible (necesita 1m + trades + iniciales).")

# =========================================
# (6) 🧾 Trades (con PnL y duración)
# =========================================
st.subheader("🧾 Trades (con PnL y duración)")
if trades_df is not None and not trades_df.empty:
    t_enr = enrich_trades_fifo(trades_df)
    if "pnl_usdt" in t_enr.columns:
        t_enr["pnl_usdt_cum"] = t_enr["pnl_usdt"].fillna(0).cumsum()
    st.dataframe(t_enr.tail(200), use_container_width=True)
else:
    st.info("Sin operaciones registradas todavía.")

# ============================================================
# (7) 📊 Distribución de acciones ejecutadas (últimos N trades)
# ============================================================
st.subheader(f"📊 Distribución de acciones ejecutadas (últimos {dist_n} trades)")
if trades_df is not None and not trades_df.empty:
    t = trades_df.copy()
    if "symbol" in t.columns:
        t = t[t["symbol"] == SYMBOL]
    t = t.sort_values("timestamp").tail(dist_n)
    if not t.empty and "side" in t.columns:
        counts = t["side"].value_counts().reindex(["buy","sell","hold"], fill_value=0)
        c1, c2 = st.columns(2)
        with c1:
            st.bar_chart(counts)
        with c2:
            pie_df = counts.reset_index()
            pie_df.columns = ["side","count"]
            fig_pie = px.pie(pie_df, names="side", values="count", title="Proporción")
            fig_pie.update_layout(margin=dict(l=10, r=10, t=35, b=10), height=300)
            st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("No hay suficientes trades para calcular la distribución.")
else:
    st.info("Sin trades aún.")

# =========================================
# (8) 📁 Estado de datos y modelo
# =========================================
st.subheader("📁 Estado de datos y modelo")

def csv_status(tf: str):
    path = os.path.join(DATA_DIR, f"ohlcv_{SYMBOL}_{tf}.csv")
    exists = os.path.exists(path)
    last_file_mtime = human_mtime(path) if exists else None
    last_row_ts = None
    nrows = None
    if exists:
        try:
            dfl = pd.read_csv(path)
            nrows = len(dfl)
            last_row_ts = pd.to_datetime(dfl["timestamp"].iloc[-1], utc=True, errors="coerce") \
                            .tz_convert(TZ_LOCAL).strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            pass
    return {"tf": tf, "csv": path, "estado": "✅ OK" if exists else "⏳ Pendiente",
            "mtime_fichero": last_file_mtime, "ultima_vela": last_row_ts, "filas": nrows}

rows = [csv_status(tf) for tf in ["1m","5m","15m","1h"]]
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

model_candidates = [
    os.path.join(DATA_DIR, "ai_model.pkl"),
    os.path.join(DATA_DIR, "ai_signal_model.pkl"),
    os.path.join(DATA_DIR, "ai_feature_names.json"),
]
present = [p for p in model_candidates if os.path.exists(p)]

if present:
    for p in present:
        st.write(f"• `{os.path.basename(p)}` → mtime: **{human_mtime(p)}**")
else:
    st.write("Modelo/artefactos IA no encontrados aún.")

st.caption(f"Auto-actualiza cada ~{refresh_sec}s. Horarios mostrados en Europe/Madrid.")


st.subheader("🧪 Tests directos de API (opcional)")

c1, c2 = st.columns(2)
with c1:
    st.caption("POST /signal/stockformer")
    if st.button("Probar Stockformer (POST)"):
        payload = {"symbol": SYMBOL, "timeframes": ["1m","5m","15m","1h"], "features": {"1m":[], "5m":[], "15m":[], "1h":[]}}
        st.json(requests.post(f"{API_BASE}/signal/stockformer", json=payload, timeout=8).json())

with c2:
    st.caption("POST /agent/multimodal/action")
    if st.button("Probar RL Multimodal (POST)"):
        payload = {"symbols":[SYMBOL], "price_window":[[100,100.2,100.1,100.6,100.9]], "sentiment": 0.2}
        st.json(requests.post(f"{API_BASE}/agent/multimodal/action", json=payload, timeout=8).json())

st.subheader("🧰 Noticias & Sentimiento — acciones rápidas")
c1, c2, c3 = st.columns([1,1,2])

with c1:
    if st.button("📥 Ingestar noticias (RSS)"):
        with st.spinner("Ingeriendo fuentes…"):
            res = safe_post_json(f"{API_BASE}/data/news/ingest")
        if isinstance(res, dict) and res.get("ok"):
            st.success(f"Ingesta OK · {res.get('rows','?')} titulares")
            try:
                st.rerun()
            except Exception:
                st.experimental_rerun()
        else:
            st.error(f"Fallo ingesta: {res.get('error',res)}")

with c2:
    if st.button("🧪 Construir sentimiento"):
        with st.spinner("Calculando sentimiento (FinBERT + decaimiento)…"):
            res = safe_post_json(f"{API_BASE}/data/sentiment/build")
        if isinstance(res, dict) and res.get("ok"):
            st.success(f"Sentimiento OK · días={res.get('rows','?')} · drivers={res.get('drivers_rows','?')}")
            try:
                st.rerun()
            except Exception:
                st.experimental_rerun()
        else:
            st.error(f"Fallo sentimiento: {res.get('error',res)}")


# =========================================
# (9) 📰 Sentimiento de noticias + enlaces
# =========================================
st.subheader("📰 Sentimiento de noticias")

# Serie de sentimiento (últimos 200)
sent = safe_request_json(f"{API_BASE}/data/sentiment/{SYMBOL}?last_n=200")
if isinstance(sent, dict) and "rows" in sent:
    sdf = pd.DataFrame(sent["rows"])
    if not sdf.empty:
        sdf["date"] = pd.to_datetime(sdf["date"], errors="coerce")
        fig_sent = px.line(sdf, x="date", y="sentiment", title="Sentimiento diario (ValueEmbedding ponderado)")
        fig_sent.update_layout(margin=dict(l=10,r=10,t=35,b=10), height=260, xaxis=dict(autorange=True), yaxis=dict(autorange=True))
        st.plotly_chart(fig_sent, use_container_width=True)
    else:
        st.info("No hay datos de sentimiento aún.")
else:
    st.warning(sent.get("error", "No se pudo cargar el sentimiento") if isinstance(sent, dict) else sent)

# Drivers (top noticias que más han contribuido recientemente)
st.caption("Noticias que más han empujado el sentimiento (positivo y negativo):")
drv = safe_request_json(f"{API_BASE}/data/sentiment/drivers/{SYMBOL}?last_n=20")
if isinstance(drv, dict) and "rows" in drv:
    ddf = pd.DataFrame(drv["rows"])
    if not ddf.empty:
        # Lista rápida con enlaces clicables (top 10)
        st.markdown("**Top recientes:**")
        for _, r in ddf.head(10).iterrows():
            date_str = str(r.get("date","-"))
            src = str(r.get("source","-"))
            score = float(r.get("ve_w", 0.0))
            title = str(r.get("headline","(sin título)"))
            url = str(r.get("url",""))
            st.markdown(f"- {date_str} · **{src}** · {score:+.3f} · [{title}]({url})")
        # Tabla completa (con columnas útiles)
        show_cols = ["date","direction","ve_w","source","headline","url"]
        st.dataframe(ddf[show_cols], use_container_width=True, hide_index=True)
    else:
        st.info("Sin drivers aún. Ejecuta la ingesta y construcción de sentimiento.")
else:
    st.warning(drv.get("error", "No se pudo cargar drivers") if isinstance(drv, dict) else drv)
