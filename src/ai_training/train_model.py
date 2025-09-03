import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

DATA_DIR = os.getenv("DATA_DIR", "./data")
SYMBOL = os.getenv("SYMBOL", "BTCUSDT")
MODEL_PATH = os.path.join(DATA_DIR, f"model_{SYMBOL}.pkl")

def load_dataset(symbol: str) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, f"dataset_{symbol}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontró dataset: {path}")
    return pd.read_csv(path)

def train_model(df: pd.DataFrame):
    feature_cols = ["rsi", "ema20", "ema50", "vol_ratio"]
    X = df[feature_cols]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("📊 Reporte de clasificación:")
    print(classification_report(y_test, y_pred))

    return model

if __name__ == "__main__":
    df = load_dataset(SYMBOL)
    model = train_model(df)

    joblib.dump(model, MODEL_PATH)
    print(f"✅ Modelo entrenado guardado en {MODEL_PATH}")
