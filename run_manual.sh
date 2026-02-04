#!/bin/bash
set -e

# Manual pipeline using the Kaggle PlantVillage dataset.
# Requires Kaggle credentials and dataset terms acceptance.

KAGGLE_DATASET="emmarex/plantdisease"
IMAGE_DIR="./kaggle_images"
CATALOG="crop_catalog.csv"
PROCESSED_DIR="./processed"
MODEL_DIR="./models"
PREDICTIONS="predictions.json"
ACCURACY="accuracy_results.json"
REPORT_DIR="./report"

echo "=== Step 1: Download and catalog images from Kaggle ==="
./fetch_crop_images.py \
    --source kaggle \
    --dataset "${KAGGLE_DATASET}" \
    --output "${CATALOG}" \
    --output-dir "${IMAGE_DIR}"

echo "=== Step 2: Preprocess images ==="
./bin/preprocess_images.py \
    --input "${CATALOG}" \
    --output-dir "${PROCESSED_DIR}" \
    --image-size 224 \
    --split 0.8

echo "=== Step 3: Train classifier ==="
./bin/train_classifier.py \
    --input-dir "${PROCESSED_DIR}" \
    --output-dir "${MODEL_DIR}" \
    --epochs 20 \
    --batch-size 32

echo "=== Step 4: Classify images ==="
./bin/classify_disease.py \
    --model-dir "${MODEL_DIR}" \
    --input "${IMAGE_DIR}" \
    --output "${PREDICTIONS}"

echo "=== Step 5: Evaluate accuracy ==="
./bin/evaluate_accuracy.py \
    --predictions "${PREDICTIONS}" \
    --catalog "${CATALOG}" \
    --output "${ACCURACY}"

echo "=== Step 6: Generate report ==="
./bin/generate_report.py \
    --predictions "${PREDICTIONS}" \
    --output-dir "${REPORT_DIR}" \
    --format all \
    --accuracy "${ACCURACY}"

echo "=== Complete! Report at ${REPORT_DIR}/report.html ==="
