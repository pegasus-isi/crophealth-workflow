#!/bin/bash
#
# Crop Health Workflow - Example Usage
#
# This script demonstrates how to use the crop health workflow
# for disease detection and classification.

# ==============================================================================
# Prerequisites
# ==============================================================================
#
# 1. Install Python dependencies:
#    pip install pandas numpy matplotlib scipy pillow scikit-learn torch requests
#
# 2. (Optional) For Kaggle data source:
#    pip install kagglehub
#    export KAGGLE_USERNAME="your-username"
#    export KAGGLE_KEY="your-api-key"
#    # Accept dataset terms: https://www.kaggle.com/datasets/emmarex/plantdisease
#
# 3. Build the Apptainer container (from the workflow root, no registry push):
#    apptainer build Apptainer/CropHealth_Container.sif \
#        Apptainer/CropHealth_Container.def
#
#    A .sif has no multi-arch manifest (one file, one architecture), so DPU/edge
#    runs need a second .sif built on an aarch64 host, passed via
#    --container-sif. Apptainer cannot build on macOS; see APPTAINER.md.

# ==============================================================================
# Step 1: Organize Local Images
# ==============================================================================

echo "=== Downloading from Kaggle ==="
./fetch_crop_images.py \
    --source kaggle \
    --dataset emmarex/plantdisease \
    --output crop_catalog.csv \
    --output-dir ./kaggle_images

# ==============================================================================
# Step 2: Or Use Sample Data for Testing
# ==============================================================================

echo "=== Skipping sample catalog (using Kaggle data) ==="

# ==============================================================================
# Step 3: Or Download from Kaggle (requires API key)
# ==============================================================================

# (Optional) If you want to use local images instead:
# ./fetch_crop_images.py \
#     --source local \
#     --input-dir ./field_images \
#     --output crop_catalog.csv \
#     --output-dir ./local_images

# ==============================================================================
# Step 4: Preprocess Images
# ==============================================================================

echo "=== Preprocessing images ==="
./bin/preprocess_images.py \
    --input crop_catalog.csv \
    --output-dir ./processed \
    --image-size 224 \
    --split 0.8

# ==============================================================================
# Step 5: Train Disease Classifier
# ==============================================================================

echo "=== Training classifier ==="
./bin/train_classifier.py \
    --input-dir ./processed \
    --output-dir ./models \
    --epochs 20 \
    --batch-size 32

# ==============================================================================
# Step 6: Run Inference on Images
# ==============================================================================

echo "=== Classifying images ==="
./bin/classify_disease.py \
    --model-dir ./models \
    --input ./kaggle_images \
    --output predictions.json

# ==============================================================================
# Step 7: Generate Health Report
# ==============================================================================

echo "=== Generating report ==="
./bin/generate_report.py \
    --predictions predictions.json \
    --output-dir ./report \
    --format all

echo "=== Report generated at ./report/report.html ==="

# ==============================================================================
# Generate Pegasus Workflow (Standard Mode)
# ==============================================================================

echo "=== Generating standard workflow ==="

./workflow_generator.py \
    --data-source kaggle \
    --kaggle-dataset emmarex/plantdisease \
    --image-size 224 \
    --epochs 20 \
    --batch-size 32 \
    --output workflow.yml

# Submit to HTCondor
# pegasus-plan --submit -s condorpool -o local workflow.yml

# ==============================================================================
# Generate Pegasus Workflow (Edge-to-Cloud DPU Mode)
# ==============================================================================

echo "=== Generating edge-to-cloud workflow ==="

./workflow_generator.py \
    --data-source kaggle \
    --kaggle-dataset emmarex/plantdisease \
    --image-size 224 \
    --epochs 20 \
    --enable-dpu \
    --edge-site edgepool \
    --cloud-site cloudpool \
    --output workflow_dpu.yml

# Submit edge-to-cloud workflow
# pegasus-plan --submit -s edgepool -s cloudpool -o local workflow_dpu.yml

# ==============================================================================
# Workflow with Local Images
# ==============================================================================

echo "=== Generating workflow for local images ==="

./workflow_generator.py \
    --data-source local \
    --image-dir ./field_images \
    --image-size 224 \
    --epochs 30 \
    --output workflow_local.yml

# ==============================================================================
# Workflow with Kaggle Data
# ==============================================================================

# ./workflow_generator.py \
#     --data-source kaggle \
#     --kaggle-dataset emmarex/plantdisease \
#     --epochs 50 \
#     --output workflow_kaggle.yml

# ==============================================================================
# Monitor Workflow
# ==============================================================================

# pegasus-status <run_directory>
# pegasus-analyzer <run_directory>

# ==============================================================================
# Supported Diseases
# ==============================================================================
#
# Apple:    Apple Scab, Black Rot, Cedar Apple Rust
# Corn:     Gray Leaf Spot, Common Rust, Northern Leaf Blight
# Grape:    Black Rot, Esca, Leaf Blight
# Potato:   Early Blight, Late Blight
# Tomato:   Bacterial Spot, Early Blight, Late Blight, Leaf Mold,
#           Septoria Leaf Spot, Spider Mites, Target Spot,
#           Yellow Leaf Curl Virus, Mosaic Virus

echo "=== Done ==="
