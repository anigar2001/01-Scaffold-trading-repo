#!/usr/bin/env bash
set -euo pipefail

echo "[trainer] $(date) Iniciando preproceso + entrenamiento"
python src/ai/data_preprocessor.py
python src/ai/train_model.py
# scripts/daily_train.sh (añade al final si quieres)
python -u src/ai_training/train_stockformer.py --data-dir /app/data --models-dir /app/models || true
python -u src/ai_training/train_multimodal.py --data-dir /app/data --models-dir /app/models || true

# (opcional) recarga suave: mover a .tmp y reemplazar atómico si construyes en otra ruta
# mv /app/data/ai_model.pkl.tmp /app/data/ai_model.pkl

echo "[trainer] $(date) Ciclo finalizado"
