# src/bot/train_lgb.py
import pandas as pd, joblib, os
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
from bot.features import add_features
from bot.features import make_labels # ensure import path

DATA_DIR = "/app/data"
MODEL_FILE = os.path.join(DATA_DIR, "lgb_model.pkl")
SCALER_FILE = os.path.join(DATA_DIR, "scaler.pkl")

if __name__ == "__main__":
    df = pd.read_csv(os.path.join(DATA_DIR, "ohlcv_BTCUSDT_1m.csv"))
    df = add_features(df)
    df = make_labels(df, horizon=5, up_th=0.0015, down_th=-0.0015)
    X = df[['close','sma_5','sma_21','ema_9','rsi_14','macd','bb_width','obv','vwap','logret']].values
    y = df['label'].values
    tscv = TimeSeriesSplit(n_splits=5)
    # simple training with last split as validation
    train_idx, val_idx = list(tscv.split(X))[-1]
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    params = {
      'objective': 'multiclass',
      'num_class': 3,
      'metric': 'multi_logloss',
      'learning_rate': 0.05,
      'num_leaves': 31,
      'verbosity': -1,
    }
    model = lgb.train(params, train_data, valid_sets=[val_data], num_boost_round=500, early_stopping_rounds=50)
    joblib.dump(model, MODEL_FILE)
    print("Saved model:", MODEL_FILE)
