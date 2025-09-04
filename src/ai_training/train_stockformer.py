# src/ai_training/train_stockformer.py
from __future__ import annotations
import os
import json
import math
import argparse
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

# PyTorch (opcional; si no está, guardamos dummy)
try:
    import torch
    from torch import nn
    _HAS_TORCH = True
except Exception:
    torch = None
    nn = None
    _HAS_TORCH = False

# =======================
# Config por entorno
# =======================
DATA_DIR = os.getenv("DATA_DIR", "/app/data")
OUT_FILE = os.getenv("STOCKFORMER_MODEL_FILE", os.path.join(DATA_DIR, "stockformer_model.pt"))
META_FILE = os.path.join(DATA_DIR, "stockformer_meta.json")

# =======================
# Utilidades de datos
# =======================
def _load_csv(symbol: str, tf: str) -> Optional[pd.DataFrame]:
    path = os.path.join(DATA_DIR, f"ohlcv_{symbol}_{tf}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    if "close" not in df.columns or df.empty:
        return None
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["close"])
    return df

def _log_returns_from_close(close: np.ndarray) -> np.ndarray:
    logp = np.log(close + 1e-12, dtype=np.float64)
    rets = np.diff(logp).astype(np.float32)
    return rets  # len = N-1

def _reindex_to_1m(base_ts: np.ndarray, ts_src: np.ndarray, vals_src: np.ndarray) -> np.ndarray:
    """
    Lleva una serie (en su propio índice) al índice 1m (forward-fill).
    base_ts: timestamps 1m (len=N)
    ts_src: timestamps de la serie fuente (len=M)
    vals_src: valores (len=M)
    Devuelve np.array len=N-1, alineado con retornos 1m.
    """
    s = pd.Series(vals_src, index=pd.Index(ts_src, name="ts"))
    s1 = s.reindex(pd.Index(base_ts, name="ts"), method="ffill")
    a = s1.to_numpy()
    # Para usar como "retorno por minuto", repetimos valor por minuto entre cierres de su TF
    # y alineamos a longitud N-1 (misma que returns 1m)
    if a.shape[0] >= 2:
        a = a[1:]  # alinear con retornos (entre t-1 y t)
    return np.nan_to_num(a.astype(np.float32), nan=0.0)

def _derive_tf_from_1m(rets_1m: np.ndarray, step: int) -> np.ndarray:
    """
    Deriva retornos de un TF (ej. 5m, 15m, 60m) desde 1m sumando 'step' retornos log.
    Devuelve serie "por minuto" forward-filled: para cada minuto toma el último retorno de bloque.
    """
    if rets_1m.size < step:
        return np.zeros_like(rets_1m, dtype=np.float32)
    kernel = np.ones(step, dtype=np.float64)
    # r_step en puntos donde cierra la vela 'step'
    r_step_edges = np.convolve(rets_1m, kernel, mode="valid").astype(np.float32)  # len = N-1-(step-1)
    # Expandimos a 'por minuto' con forward-fill
    pad = np.full(step - 1, np.nan, dtype=np.float32)
    aligned = np.concatenate([pad, r_step_edges])
    # Forward-fill NaNs para llegar a len = N-1
    m = pd.Series(aligned).ffill().fillna(0.0).to_numpy(dtype=np.float32)
    return m

def _sliding_windows(x: np.ndarray, L: int) -> np.ndarray:
    """
    Ventanas deslizantes (N-L+1, L) sobre un vector x (N,).
    """
    from numpy.lib.stride_tricks import sliding_window_view
    if x.size < L:
        # Devuelve 0 filas si no hay suficiente longitud
        return np.zeros((0, L), dtype=np.float32)
    return sliding_window_view(x, L)

def _build_dataset_multitf(
    df_1m: pd.DataFrame,
    df_5m: Optional[pd.DataFrame],
    df_15m: Optional[pd.DataFrame],
    df_1h: Optional[pd.DataFrame],
    lookbacks: Dict[str, int],
    horizon_min: int,
    buy_th: float,
    sell_th: float,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float], int]:
    """
    Construye X (batch, seq_len, 1) y y (batch,) con etiqueta a H min.
    Retorna también los escalados por TF y la seq_len total.
    """
    # Base 1m
    ts1 = df_1m["timestamp"].to_numpy()
    c1 = df_1m["close"].to_numpy(dtype=np.float64)
    rets1 = _log_returns_from_close(c1)                  # len = N-1
    N = rets1.shape[0]

    # Retornos por TF real, reindexados a 1m. Si no hay CSV, derivamos de 1m.
    def tf_series(df_tf: Optional[pd.DataFrame], step: int) -> np.ndarray:
        if df_tf is not None:
            c = df_tf["close"].to_numpy(dtype=np.float64)
            ts = df_tf["timestamp"].to_numpy()
            r = _log_returns_from_close(c)  # en su propio índice
            # Serie de "retorno por barra TF"; la llevamos a índice 1m con ffill
            # Primero creamos timestamps de r que empiezan en ts[1:]
            vals = r
            ts_vals = ts[1:]
            return _reindex_to_1m(ts1, ts_vals, vals)
        else:
            return _derive_tf_from_1m(rets1, step)

    r1m = rets1.copy()
    r5m = tf_series(df_5m, step=5)
    r15m = tf_series(df_15m, step=15)
    r60m = tf_series(df_1h, step=60)

    # Normalización por TF (dividir por 5*std)
    def _scale(x: np.ndarray) -> Tuple[np.ndarray, float]:
        std = float(np.std(x))
        scale = (5.0 * std + 1e-8) if std > 0 else 1.0
        return (x / scale).astype(np.float32), scale

    r1m_n, s1 = _scale(r1m)
    r5m_n, s5 = _scale(r5m)
    r15m_n, s15 = _scale(r15m)
    r60m_n, s60 = _scale(r60m)

    L1 = int(lookbacks.get("1m", 32))
    L5 = int(lookbacks.get("5m", 16))
    L15 = int(lookbacks.get("15m", 8))
    L60 = int(lookbacks.get("1h", 4))

    W1 = _sliding_windows(r1m_n, L1)   # (N - L1 + 1, L1)
    W5 = _sliding_windows(r5m_n, L5)
    W15 = _sliding_windows(r15m_n, L15)
    W60 = _sliding_windows(r60m_n, L60)

    # M = nº de filas comunes en todas las ventanas
    M = min(W1.shape[0], W5.shape[0], W15.shape[0], W60.shape[0])
    if M <= 0:
        raise RuntimeError("No hay suficientes datos para construir ventanas.")

    # Para etiquetar a H min, necesitamos H pasos futuros → recortamos al final
    H = int(horizon_min)
    M_eff = M - H
    if M_eff <= 0:
        raise RuntimeError("No hay suficientes datos tras aplicar el horizonte futuro.")

    # Tomamos las ÚLTIMAS M_eff filas para alinear con etiquetas
    W1 = W1[-M_eff:]
    W5 = W5[-M_eff:]
    W15 = W15[-M_eff:]
    W60 = W60[-M_eff:]

    # Construye secuencia por muestra concatenando [1m,5m,15m,1h] → seq_len = L1+L5+L15+L60
    seq_len = L1 + L5 + L15 + L60
    X = np.concatenate([W1, W5, W15, W60], axis=1).astype(np.float32)  # (M_eff, seq_len)

    # Etiquetas: fwd ret a H min desde logp (índice de precio)
    logp = np.log(c1 + 1e-12)
    fwd = (logp[H:] - logp[:-H]).astype(np.float32)                    # len = Np - H
    # Índice de precio donde termina cada ventana de retornos:
    # para la fila j en 0..M_eff-1, p_end = (N - M) + j + 1
    p_end_start = (N - M) + 1
    p_end_idx = np.arange(p_end_start, p_end_start + M_eff, dtype=np.int64)
    fwd_sel = fwd[p_end_idx]  # selecciona H-min forward ret por muestra

    y = np.zeros(M_eff, dtype=np.int64)
    y[fwd_sel > buy_th] = 1
    y[fwd_sel < sell_th] = -1
    # Mapear a {0,1,2} para CE: {-1,0,1} -> {0,1,2}
    y_ce = (y + 1).astype(np.int64)

    # Devuelve X como (batch, seq_len, 1)
    X3 = X[..., None]  # añade canal
    scales = {"1m": s1, "5m": s5, "15m": s15, "1h": s60}
    return X3, y_ce, scales, seq_len

# =======================
# Modelo tipo Stockformer (mini)
# =======================
if _HAS_TORCH:
    class PositionalEncoding(nn.Module):
        def __init__(self, d_model: int, max_len: int):
            super().__init__()
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer("pe", pe.unsqueeze(1))  # (max_len,1,d_model)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (S,B,D)
            S = x.size(0)
            return x + self.pe[:S]

    class TinyStockformer(nn.Module):
        """
        Entrada: (B, S, 1) con S = sum(lookbacks). Proyección -> Transformer -> pooling -> head.
        Salida: logits (B, 3)
        """
        def __init__(self, seq_len: int, d_model: int = 64, nhead: int = 4, num_layers: int = 2, dropout: float = 0.1):
            super().__init__()
            self.seq_len = seq_len
            self.proj = nn.Linear(1, d_model)
            encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=False)
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.pos = PositionalEncoding(d_model, max_len=seq_len)
            self.head = nn.Sequential(
                nn.Linear(d_model, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 3),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (B, S, 1)
            B, S, _ = x.shape
            h = self.proj(x)               # (B,S,D)
            h = h.transpose(0, 1)          # (S,B,D) para Transformer (batch en dim 1)
            h = self.pos(h)                # (S,B,D)
            h = self.encoder(h)            # (S,B,D)
            h = h.mean(dim=0)              # (B,D)
            out = self.head(h)             # (B,3)
            return out

# =======================
# Entrenamiento
# =======================
def train(
    symbol: str = "BTCUSDT",
    horizon_min: int = 15,
    lookback_mtf: str = "32,16,8,4",
    buy_th: float = 0.002,
    sell_th: float = -0.002,
    epochs: int = 5,
    batch_size: int = 256,
    lr: float = 1e-3,
    val_frac: float = 0.2,
    seed: int = 42,
) -> None:
    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)

    # Carga datos
    df_1m = _load_csv(symbol, "1m")
    if df_1m is None:
        raise RuntimeError(f"Falta CSV 1m: {os.path.join(DATA_DIR, f'ohlcv_{symbol}_1m.csv')}")
    df_5m = _load_csv(symbol, "5m")
    df_15m = _load_csv(symbol, "15m")
    df_1h = _load_csv(symbol, "1h")

    Ls = [int(x.strip()) for x in lookback_mtf.split(",")] if lookback_mtf else [32, 16, 8, 4]
    lookbacks = {"1m": Ls[0], "5m": Ls[1], "15m": Ls[2], "1h": Ls[3]}

    # Construye dataset
    X, y, scales, seq_len = _build_dataset_multitf(
        df_1m=df_1m, df_5m=df_5m, df_15m=df_15m, df_1h=df_1h,
        lookbacks=lookbacks, horizon_min=horizon_min, buy_th=buy_th, sell_th=sell_th
    )

    # Downsample opcional para ahorrar RAM/tiempo (STRIDE=1 no cambia nada)
    STRIDE = int(os.getenv("STOCKFORMER_STRIDE", "1"))
    if STRIDE > 1:
        X = X[::STRIDE]
        y = y[::STRIDE]

    # Partición temporal: train (inicio) / val (final)
    N = X.shape[0]
    n_val = max(1, int(N * val_frac))
    n_tr = N - n_val
    X_tr, y_tr = X[:n_tr], y[:n_tr]
    X_val, y_val = X[n_tr:], y[n_tr:]

    if not _HAS_TORCH:
        # Dummy si no hay Torch
        with open(OUT_FILE, "w") as f:
            f.write("DUMMY_STOCKFORMER\n")
        meta = {
            "symbol": symbol,
            "tfs": ["1m", "5m", "15m", "1h"],
            "lookbacks": lookbacks,
            "horizon_min": horizon_min,
            "buy_th": buy_th,
            "sell_th": sell_th,
            "seq_len": seq_len,
            "scales": scales,
            "backend": "dummy",
        }
        with open(META_FILE, "w") as f:
            json.dump(meta, f)
        print(f"[OK] Guardado dummy en {OUT_FILE}")
        return

    # Semillas
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cpu")

    # Tamaño de modelo configurable por ENV (para VPS con poca RAM baja estos valores)
    DMODEL = int(os.getenv("STOCKFORMER_DMODEL", "64"))
    NHEAD = int(os.getenv("STOCKFORMER_NHEAD", "4"))
    LAYERS = int(os.getenv("STOCKFORMER_LAYERS", "2"))

    model = TinyStockformer(seq_len=seq_len, d_model=DMODEL, nhead=NHEAD, num_layers=LAYERS, dropout=0.1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    # Pondera clases si hay desequilibrio
    cls_counts = np.bincount(y_tr, minlength=3).astype(np.float32)
    inv = 1.0 / np.clip(cls_counts, 1.0, None)
    w = inv / inv.sum() * 3.0
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32))

    def _batches(Xa: np.ndarray, ya: np.ndarray, bs: int):
        for i in range(0, Xa.shape[0], bs):
            yield Xa[i:i+bs], ya[i:i+bs]

    # Entrenamiento
    model.train()
    for ep in range(1, epochs + 1):
        tr_loss = 0.0
        tr_correct = 0
        tr_total = 0
        for xb, yb in _batches(X_tr, y_tr, batch_size):
            xb_t = torch.from_numpy(xb).to(device)
            yb_t = torch.from_numpy(yb).to(device)
            opt.zero_grad()
            logits = model(xb_t)         # (B,3)
            loss = criterion(logits, yb_t)
            loss.backward()
            opt.step()
            tr_loss += float(loss.item()) * xb.shape[0]
            pred = logits.argmax(dim=1).cpu().numpy()
            tr_correct += int((pred == yb).sum())
            tr_total += int(xb.shape[0])

        # ---- Validación por mini-batches (evita OOM) ----
        val_bs = min(batch_size, 512)
        val_loss_accum = 0.0
        val_correct = 0
        val_count = 0
        model.eval()
        with torch.no_grad():
            for xb, yb in _batches(X_val, y_val, val_bs):
                xb_t = torch.from_numpy(xb).to(device)
                yb_t = torch.from_numpy(yb).to(device)
                logits_val = model(xb_t)
                val_loss_accum += float(nn.functional.cross_entropy(logits_val, yb_t, reduction="sum").item())
                val_correct += int((logits_val.argmax(dim=1) == yb_t).sum().item())
                val_count += int(xb.shape[0])
        val_loss = val_loss_accum / max(1, val_count)
        val_acc = val_correct / max(1, val_count)
        model.train()
        # -----------------------------------------------

        tr_acc = tr_correct / max(1, tr_total)
        print(f"[Epoch {ep}/{epochs}] loss={tr_loss/max(1,tr_total):.4f} "
              f"val_loss={val_loss:.4f} tr_acc={tr_acc:.3f} val_acc={val_acc:.3f}")

    # Export TorchScript (eval)
    model.eval()
    ex = torch.from_numpy(X[:1]).to(device)  # (1, S, 1)
    with torch.no_grad():
        scripted = torch.jit.trace(model, ex)

    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    scripted.save(OUT_FILE)

    meta = {
        "symbol": symbol,
        "tfs": ["1m", "5m", "15m", "1h"],
        "lookbacks": lookbacks,
        "horizon_min": horizon_min,
        "buy_th": buy_th,
        "sell_th": sell_th,
        "seq_len": seq_len,
        "scales": {k: float(v) for k, v in scales.items()},
        "backend": "torch",
        "notes": "inputs: concat ventanas [1m,5m,15m,1h] de retornos log normalizados; salida logits [sell, hold, buy]",
        "model_cfg": {"d_model": DMODEL, "nhead": NHEAD, "layers": LAYERS},
        "stride": STRIDE,
    }
    with open(META_FILE, "w") as f:
        json.dump(meta, f)

    print(f"[OK] Guardado modelo Stockformer en {OUT_FILE}")
    print(f"[OK] Meta en {META_FILE}")


# =======================
# CLI
# =======================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Entrena Stockformer (multi-TF) para objetivo 15 min.")
    p.add_argument("--symbol", type=str, default="BTCUSDT")
    p.add_argument("--horizon-min", type=int, default=int(os.getenv("STOCKFORMER_HORIZON_MIN", "15")))
    p.add_argument("--lookback-mtf", type=str, default=os.getenv("STOCKFORMER_LOOKBACK_MTF", "32,16,8,4"),
                   help="Lookbacks por TF: 1m,5m,15m,1h (p.ej. '32,16,8,4')")
    p.add_argument("--buy-th", type=float, default=0.002)
    p.add_argument("--sell-th", type=float, default=-0.002)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train(
        symbol=args.symbol,
        horizon_min=args.horizon_min,
        lookback_mtf=args.lookback_mtf,
        buy_th=args.buy_th,
        sell_th=args.sell_th,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_frac=args.val_frac,
        seed=args.seed,
    )
