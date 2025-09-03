import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os

DATA_DIR = "/app/data"
CSV_FILE = os.path.join(DATA_DIR, "BTCUSDT_ohlcv.csv")
MODEL_FILE = os.path.join(DATA_DIR, "ai_signal_model.pkl")
SCALER_FILE = os.path.join(DATA_DIR, "scaler.pkl")

# Verificar que el CSV existe
if not os.path.exists(CSV_FILE):
    raise FileNotFoundError(f"No se encontró {CSV_FILE}. Descarga los datos primero.")

# Leer datos históricos
df = pd.read_csv(CSV_FILE, parse_dates=['timestamp'])

# Crear columna futura para la señal
df['future_close'] = df['close'].shift(-1)
df['signal'] = (df['future_close'] > df['close']).astype(int)
df.dropna(inplace=True)

# Crear features simples
df['hl'] = df['high'] - df['low']
df['oc'] = df['open'] - df['close']
X = df[['close', 'hl', 'oc', 'volume']].values
y = df['signal'].values

# Escalar features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Entrenar modelo RandomForest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_scaled, y)

# Guardar modelo y scaler
joblib.dump(model, MODEL_FILE)
joblib.dump(scaler, SCALER_FILE)

print(f"Modelo de IA entrenado y guardado en {MODEL_FILE}")
