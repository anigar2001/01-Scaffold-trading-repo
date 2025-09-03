# src/ai/train_model.py

import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import List, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import TimeSeriesSplit


# ==========================
# Configuración
# ==========================
DATA_DIR = os.getenv("DATA_DIR", "/app/data")
COMBINED_FILE = os.path.join(DATA_DIR, "combined_features.pkl")
MODEL_FILE = os.path.join(DATA_DIR, "ai_model.pkl")
FEATURES_FILE = os.path.join(DATA_DIR, "ai_feature_names.json")

# Etiquetado
HORIZON = int(os.getenv("LABEL_HORIZON", "5"))         # barras hacia delante 5 (en 1m si tu índice base es 1m)
UP_TH = float(os.getenv("LABEL_UP_TH", "0.002"))        # +0.2%
DOWN_TH = float(os.getenv("LABEL_DOWN_TH", "-0.002"))   # -0.2%

# Modelo
N_SPLITS = int(os.getenv("TS_SPLITS", "5"))
RF_TREES = int(os.getenv("RF_TREES", "300"))
RF_MAX_DEPTH = int(os.getenv("RF_MAX_DEPTH", "12"))

# Prefijos de columnas (de tu data_preprocessor)
TF_PREFIXES = ("t1m_", "t5m_", "t15m_", "t1h_")  # ¡deben coincidir con tu preproceso!


# ==========================
# Funciones auxiliares
# ==========================
def load_combined_dataset(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontró el dataset combinado: {path}\n"
                                f"Genera primero con: src/ai/data_preprocessor.py")
    df = joblib.load(path)
    if not isinstance(df, pd.DataFrame):
        raise TypeError("El archivo combinado no es un DataFrame.")
    # Asegurar orden temporal
    df = df.sort_index()
    return df


def create_labels(df: pd.DataFrame,
                  horizon: int = HORIZON,
                  up_th: float = UP_TH,
                  down_th: float = DOWN_TH) -> pd.DataFrame:
    """
    Crea la columna 'signal' con:
      1 = buy  si future_return >= up_th
      0 = hold si entre (down_th, up_th)
     -1 = sell si future_return <= down_th
    future_return = (close_{t+h} - close_t) / close_t
    Nota: usamos la serie de close del timeframe base (1m).
    """
    dfl = df.copy()
    # Busca una columna 'close' base; si no, usa la primera 'close' disponible
    close_cols = [c for c in dfl.columns if c == "close" or c.endswith("_close") or c.startswith("close")]
    if "close" in dfl.columns:
        close_series = dfl["close"].astype(float)
    elif len(close_cols) > 0:
        close_series = dfl[close_cols[0]].astype(float)
    else:
        raise ValueError("No se encontró columna 'close' en el dataset combinado.")

    dfl["future_close"] = close_series.shift(-horizon)
    dfl["pct_change"] = (dfl["future_close"] - close_series) / close_series

    dfl["signal"] = 0
    dfl.loc[dfl["pct_change"] >= up_th, "signal"] = 1
    dfl.loc[dfl["pct_change"] <= down_th, "signal"] = -1

    dfl = dfl.dropna(subset=["future_close", "pct_change"])
    return dfl


def select_feature_columns(df: pd.DataFrame,
                           prefixes: Tuple[str, ...] = TF_PREFIXES) -> List[str]:
    """
    Selecciona SOLO columnas de features que empiecen por los prefijos definidos
    (e.g., t1m_, t5m_, t15m_, t1h_) para asegurar consistencia con producción.
    """
    feature_cols = [c for c in df.columns if c.startswith(prefixes)]
    if not feature_cols:
        raise ValueError(
            "No se encontraron columnas de features con los prefijos "
            f"{prefixes}. Revisa tu data_preprocessor.py y que hayas guardado combined_features.pkl correctamente."
        )
    return feature_cols


def time_series_split_train(X: np.ndarray, y: np.ndarray, n_splits: int = N_SPLITS):
    """
    Usa TimeSeriesSplit para validar sin fuga temporal.
    Devuelve (modelo_entrenado, reporte_último_split).
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    last_report = None
    model = RandomForestClassifier(
        n_estimators=RF_TREES,
        max_depth=RF_MAX_DEPTH,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )

    split_id = 0
    for train_idx, val_idx in tscv.split(X):
        split_id += 1
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_val)

        print(f"\n=== TimeSeriesSplit {split_id}/{n_splits} ===")
        print("Distribución train:", {int(k): int(v) for k, v in pd.Series(y_tr).value_counts().items()})
        print("Distribución val  :", {int(k): int(v) for k, v in pd.Series(y_val).value_counts().items()})
        print("Matriz de confusión (val):")
        print(confusion_matrix(y_val, y_pred, labels=[-1, 0, 1]))
        print("Reporte clasificación (val):")
        print(classification_report(y_val, y_pred, digits=4))

        last_report = classification_report(y_val, y_pred, output_dict=True)

    # Reentrena en TODO el set para el modelo final de producción
    model.fit(X, y)
    return model, last_report


# ==========================
# Entrenamiento principal
# ==========================
def main():
    print("🔧 Cargando dataset combinado...")
    df = load_combined_dataset(COMBINED_FILE)
    print(f"Dataset: {COMBINED_FILE} | shape={df.shape}")

    print("🏷️  Generando etiquetas (buy/hold/sell)...")
    df_labeled = create_labels(df, HORIZON, UP_TH, DOWN_TH)
    print("Distribución de señales:", {int(k): int(v) for k, v in df_labeled["signal"].value_counts().items()})

    print("🧱 Seleccionando columnas de features por prefijo...")
    feature_cols = select_feature_columns(df_labeled, TF_PREFIXES)

    # Quitar columnas de objetivo / auxiliares si quedaron con prefijo (por seguridad)
    for col_to_drop in ["future_close", "pct_change", "signal"]:
        if col_to_drop in feature_cols:
            feature_cols.remove(col_to_drop)

    X = df_labeled[feature_cols].astype(float)
    y = df_labeled["signal"].astype(int)

    print(f"🔎 Nº de features seleccionadas: {len(feature_cols)}")
    print(f"📈 Nº de muestras: {len(y)}")

    print("🧪 Validación con TimeSeriesSplit y entrenamiento final...")
    model, last_report = time_series_split_train(X, y, n_splits=N_SPLITS)

    # Guardar artefactos
    os.makedirs(DATA_DIR, exist_ok=True)
    joblib.dump(model, MODEL_FILE)
    with open(FEATURES_FILE, "w") as f:
        json.dump(feature_cols, f)

    print(f"\n✅ Modelo guardado en: {MODEL_FILE}")
    print(f"✅ Lista de features guardada en: {FEATURES_FILE}")

    if last_report is not None:
        # Resumen corto
        try:
            acc = last_report["accuracy"]
            print(f"📊 Último split - accuracy: {acc:.4f}")
        except Exception:
            pass


if __name__ == "__main__":
    main()
