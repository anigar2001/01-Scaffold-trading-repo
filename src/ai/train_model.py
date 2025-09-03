# src/ai/train_model.py

import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Tuple, List
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import TimeSeriesSplit
from sklearn.calibration import CalibratedClassifierCV
from lightgbm import LGBMClassifier

# ==========================
# Configuración
# ==========================
DATA_DIR = os.getenv("DATA_DIR", "/app/data")
COMBINED_FILE = os.path.join(DATA_DIR, "combined_features.pkl")
MODEL_FILE = os.path.join(DATA_DIR, "ai_model.pkl")
FEATURES_FILE = os.path.join(DATA_DIR, "ai_feature_names.json")

# Etiquetado (ajustados para menos HOLD y más variación)
HORIZON = int(os.getenv("LABEL_HORIZON", "10"))      # barras hacia delante (índice base 1m)
UP_TH   = float(os.getenv("LABEL_UP_TH", "0.0008"))  # +0.08%
DOWN_TH = float(os.getenv("LABEL_DOWN_TH", "-0.0008"))

# Particionado temporal
N_SPLITS = int(os.getenv("TS_SPLITS", "5"))

# Prefijos de columnas creadas por tu data_preprocessor
TF_PREFIXES = ("t1m_", "t5m_", "t15m_", "t1h_")  # deben coincidir con data_preprocessor


# ==========================
# Utilidades
# ==========================
def load_combined_dataset(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No se encontró el dataset combinado: {path}\n"
            f"Genera primero con: python src/ai/data_preprocessor.py"
        )
    df = joblib.load(path)
    if not isinstance(df, pd.DataFrame):
        raise TypeError("El archivo combinado no es un DataFrame.")
    return df.sort_index()


def create_labels(df: pd.DataFrame,
                  horizon: int = HORIZON,
                  up_th: float = UP_TH,
                  down_th: float = DOWN_TH) -> pd.DataFrame:
    """
    Crea 'signal' con:
      1 = buy  si future_return >= up_th
      0 = hold si (down_th < future_return < up_th)
     -1 = sell si future_return <= down_th
    """
    dfl = df.copy()
    # Usa la columna 'close' base de 1m si existe; si no, la primera que contenga 'close'
    if "close" in dfl.columns:
        close = dfl["close"].astype(float)
    else:
        close_cols = [c for c in dfl.columns if "close" in c]
        if not close_cols:
            raise ValueError("No se encontró ninguna columna que contenga 'close' para etiquetar.")
        close = dfl[close_cols[0]].astype(float)

    dfl["future_close"] = close.shift(-horizon)
    dfl["pct_change"] = (dfl["future_close"] - close) / close

    dfl["signal"] = 0
    dfl.loc[dfl["pct_change"] >= up_th, "signal"] = 1
    dfl.loc[dfl["pct_change"] <= down_th, "signal"] = -1

    dfl = dfl.dropna(subset=["future_close", "pct_change"])
    return dfl


def select_feature_columns(df: pd.DataFrame,
                           prefixes: Tuple[str, ...] = TF_PREFIXES) -> List[str]:
    feats = [c for c in df.columns if c.startswith(prefixes)]
    if not feats:
        raise ValueError(
            f"No se encontraron columnas con prefijos {prefixes}. "
            "Revisa data_preprocessor.py y que combined_features.pkl contenga esas columnas."
        )
    # Eliminar por seguridad columnas que no deben ser features
    for col in ["future_close", "pct_change", "signal"]:
        if col in feats:
            feats.remove(col)
    return feats


def print_split_metrics(y_true, y_pred, split_id, total_splits):
    print(f"\n=== TimeSeriesSplit {split_id}/{total_splits} ===")
    print("Matriz de confusión (val) [rows=true, cols=pred]:")
    print(confusion_matrix(y_true, y_pred, labels=[-1, 0, 1]))
    print("Reporte de clasificación (val):")
    print(classification_report(y_true, y_pred, digits=4))


# ==========================
# Entrenamiento
# ==========================
def main():
    print("🔧 Cargando dataset combinado...")
    df = load_combined_dataset(COMBINED_FILE)
    print(f"Dataset: {COMBINED_FILE} | shape={df.shape}")

    print("🏷️  Generando etiquetas (buy/hold/sell)...")
    df = create_labels(df, HORIZON, UP_TH, DOWN_TH)
    print("Distribución de señales:", {int(k): int(v) for k, v in df["signal"].value_counts().items()})

    print("🧱 Seleccionando columnas de features (por prefijo)...")
    feature_cols = select_feature_columns(df, TF_PREFIXES)
    print(f"Nº de features: {len(feature_cols)}")

    X = df[feature_cols].astype(float)
    y = df["signal"].astype(int)

    # Modelo base LightGBM multiclass
    base = LGBMClassifier(
        objective="multiclass",
        num_class=3,
        n_estimators=800,
        learning_rate=0.05,
        max_depth=-1,
        min_data_in_leaf=50,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    # Validación temporal + métricas
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    for i, (tr, va) in enumerate(tscv.split(X), start=1):
        X_tr, X_va = X.iloc[tr], X.iloc[va]
        y_tr, y_va = y.iloc[tr], y.iloc[va]
        base.fit(X_tr, y_tr)
        y_pred = base.predict(X_va)
        print_split_metrics(y_va, y_pred, i, N_SPLITS)

    # Entrenamiento final + calibración (isotonic) usando el último 20% para calibrar
    cut = int(len(X) * 0.8)
    X_tr, y_tr = X.iloc[:cut], y.iloc[:cut]
    X_cal, y_cal = X.iloc[cut:], y.iloc[cut:]

    print("\n🧠 Entrenando modelo base en 80% y calibrando en 20% (isotonic)...")
    base.fit(X_tr, y_tr)

    cal = CalibratedClassifierCV(base, method="isotonic", cv="prefit")
    cal.fit(X_cal, y_cal)

    # Guardar artefactos
    os.makedirs(DATA_DIR, exist_ok=True)
    joblib.dump(cal, MODEL_FILE)
    with open(FEATURES_FILE, "w") as f:
        json.dump(feature_cols, f)

    print(f"\n✅ Modelo calibrado guardado en: {MODEL_FILE}")
    print(f"✅ Features guardadas en:       {FEATURES_FILE}")


if __name__ == "__main__":
    main()
