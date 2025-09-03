# src/ai_training/train_stacking.py
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

DATA_DIR = "/app/data"
SYMBOL = "BTCUSDT"
TFS = ["1m","5m","15m","1h"]         # timeframes disponibles
BASE_TF = "1m"                      # index base para samples/labels
N_SPLITS = 5

def load_and_features(symbol, tf):
    path = os.path.join(DATA_DIR, f"ohlcv_{symbol}_{tf}.csv")
    df = pd.read_csv(path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    # features example (tú reemplaza con tu add_features)
    df['close'] = df['close'].astype(float)
    df['logret'] = np.log(df['close'] / df['close'].shift(1))
    df['ema20'] = df['close'].ewm(span=20).mean()
    df['ema50'] = df['close'].ewm(span=50).mean()
    df['vol_ma20'] = df['volume'].rolling(20).mean()
    df['vol_ratio'] = df['volume'] / (df['vol_ma20'] + 1e-9)
    df = df.dropna().reset_index(drop=True)
    return df

# 1) load features per tf
dfs = {tf: load_and_features(SYMBOL, tf) for tf in TFS}

# 2) align to base timeline using merge_asof
base = dfs[BASE_TF].copy()
base = base.sort_values("timestamp").reset_index(drop=True)
combined = base[['timestamp','close']].copy()
for tf in TFS:
    if tf == BASE_TF:
        cols = ['timestamp','logret','ema20','ema50','vol_ratio']
        src = dfs[tf][cols].rename(columns=lambda c: f"{c}_{tf}" if c!='timestamp' else c)
        combined = pd.merge_asof(combined.sort_values('timestamp'),
                                 src.sort_values('timestamp'),
                                 on='timestamp',
                                 direction='backward')
    else:
        cols = ['timestamp','logret','ema20','ema50','vol_ratio']
        src = dfs[tf][cols].rename(columns=lambda c: f"{c}_{tf}" if c!='timestamp' else c)
        combined = pd.merge_asof(combined.sort_values('timestamp'),
                                 src.sort_values('timestamp'),
                                 on='timestamp',
                                 direction='backward')

combined = combined.dropna().reset_index(drop=True)

# 3) labels (ejemplo: horizon base 5)
horizon = 5
combined['future_close'] = combined['close'].shift(-horizon)
combined['fut_ret'] = combined['future_close']/combined['close'] - 1
up_th = 0.0015; down_th = -0.0015
def lab(r):
    if r >= up_th: return "buy"
    if r <= down_th: return "sell"
    return "hold"
combined['label'] = combined['fut_ret'].apply(lambda x: lab(x) if pd.notnull(x) else np.nan)
combined = combined.dropna(subset=['label']).reset_index(drop=True)

# 4) Prepare per-tf feature matrices
tf_feature_cols = {}
for tf in TFS:
    cols = [c for c in combined.columns if c.endswith(f"_{tf}")]
    tf_feature_cols[tf] = cols

# 5) OOF preds arrays
n = len(combined)
classes = ["sell","hold","buy"]
oof_preds = {tf: np.zeros((n, len(classes))) for tf in TFS}
models = {}

tscv = TimeSeriesSplit(n_splits=N_SPLITS)

for tf in TFS:
    X = combined[tf_feature_cols[tf]].values
    y = combined['label'].values
    # scaler per tf
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    # store scaler
    os.makedirs(os.path.join(DATA_DIR,'scalers'), exist_ok=True)
    joblib.dump(scaler, os.path.join(DATA_DIR, f"scalers/scaler_{tf}.pkl"))

    # OOF loop: using RF for simplicity
    for train_idx, val_idx in tscv.split(X):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr = y[train_idx]
        # use LGB or RF
        clf = RandomForestClassifier(n_estimators=200, random_state=42)
        clf.fit(X_tr, y_tr)
        proba = clf.predict_proba(X_val)  # columns correspond to clf.classes_
        # align classes ordering
        # build proba in order [sell,hold,buy]
        proba_aligned = np.zeros((len(val_idx), len(classes)))
        for i, cls in enumerate(classes):
            if cls in clf.classes_:
                idx = list(clf.classes_).index(cls)
                proba_aligned[:, i] = proba[:, idx]
            else:
                # class missing; leave zeros
                proba_aligned[:, i] = 0.0
        oof_preds[tf][val_idx] = proba_aligned

    # retrain on full set and save model
    clf_full = RandomForestClassifier(n_estimators=300, random_state=42)
    clf_full.fit(X, y)
    models[tf] = clf_full
    joblib.dump(clf_full, os.path.join(DATA_DIR, f"models/model_{SYMBOL}_{tf}.pkl"))

# 6) Build meta X from OOF preds
meta_X = np.hstack([oof_preds[tf] for tf in TFS])
meta_y = combined['label'].values

# 7) Train meta-model
meta_clf = LogisticRegression(multi_class='multinomial', max_iter=1000)
meta_clf.fit(meta_X, meta_y)
joblib.dump(meta_clf, os.path.join(DATA_DIR, f"models/meta_model_{SYMBOL}.pkl"))

# 8) Save artifact single pkl (models + meta + feature mapping)
artifact = {
    'models': {tf: models[tf] for tf in TFS},
    'meta': meta_clf,
    'tf_feature_cols': tf_feature_cols,
    'classes': classes,
    'scalers_dir': os.path.join(DATA_DIR,'scalers'),
    'config': {'symbol': SYMBOL, 'tfs': TFS, 'base_tf': BASE_TF, 'horizon': horizon}
}
joblib.dump(artifact, os.path.join(DATA_DIR, f"stacked_{SYMBOL}.pkl"))
print("Saved stacked model:", os.path.join(DATA_DIR, f"stacked_{SYMBOL}.pkl"))
