import os, json, joblib, pandas as pd

DATA_DIR = os.getenv("DATA_DIR", "/app/data")
MODEL_PATH = os.path.join(DATA_DIR, "ai_model.pkl")
FEATS_PATH = os.path.join(DATA_DIR, "ai_feature_names.json")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"No se encontró el modelo: {MODEL_PATH}")
if not os.path.exists(FEATS_PATH):
    raise FileNotFoundError(f"No se encontró la lista de features: {FEATS_PATH}")

model = joblib.load(MODEL_PATH)
with open(FEATS_PATH, "r") as f:
    feats = json.load(f)

print("Nº features:", len(feats))
print("Ejemplos de columnas:", feats[:10])

# Importancias (si el modelo las expone: RF / GBM)
if hasattr(model, "feature_importances_"):
    imp = pd.Series(model.feature_importances_, index=feats).sort_values(ascending=False)
    print("\nTOP 20 importancias:")
    print(imp.head(20).to_string())
else:
    print("\nEl modelo no expone feature_importances_. (p. ej., LogisticRegression)")
