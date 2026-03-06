# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a **Pegasus WMS workflow** for detecting and classifying crop diseases from images using CNN-based deep learning. It targets execution on ACCESS (XSEDE) and FABRIC testbed infrastructure via HTCondor.

## Architecture

The workflow is a linear DAG with 6 jobs:

```
fetch_images → preprocess → train → classify → evaluate → report
```

- **`workflow_generator.py`** — Generates the Pegasus workflow YAML (`workflow.yml`) using `Pegasus.api`. This is the entry point for creating the workflow.
- **`fetch_crop_images.py`** — Fetches/catalogs images from local disk, Kaggle, or sample data; outputs `crop_catalog.csv` + `images.tar.gz`.
- **`bin/preprocess_images.py`** — Resizes and normalizes images; outputs `train_data.npz`, `val_data.npz`, `label_mapping.json`.
- **`bin/train_classifier.py`** — Trains PyTorch CNN (transfer learning); outputs `disease_classifier.pt`.
- **`bin/classify_disease.py`** — Runs inference; outputs `predictions.json`.
- **`bin/evaluate_accuracy.py`** — Computes per-class precision/recall/F1 and confusion matrix; outputs `accuracy_results.json`.
- **`bin/generate_report.py`** — Generates HTML report + PNG charts.

All jobs run inside a Singularity container pulled from `kthare10/crophealth:latest` (Docker Hub). The container is multi-platform (amd64/arm64).

### Edge-to-Cloud (DPU) Mode

`--enable-dpu` splits the workflow across two HTCondor pools: `edgepool` (I/O-bound fetch/preprocess) and `cloudpool` (compute-bound train/classify/evaluate/report). Jobs targeting DPU workers require HTCondor ClassAd `+has_dpu = True`.

## Common Commands

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run pipeline locally (without Pegasus)
```bash
# Full manual run using Kaggle dataset
./run_manual.sh

# Or step-by-step
./fetch_crop_images.py --source sample --output crop_catalog.csv
./bin/preprocess_images.py --input crop_catalog.csv --output-dir ./processed --image-size 128 --split 0.8
./bin/train_classifier.py --input-dir ./processed --output-dir ./models --epochs 5 --batch-size 16
./bin/classify_disease.py --model-dir ./models --input ./images --output predictions.json
./bin/evaluate_accuracy.py --predictions predictions.json --catalog crop_catalog.csv --output accuracy_results.json
./bin/generate_report.py --predictions predictions.json --output-dir ./report --format all --accuracy accuracy_results.json
```

### Generate Pegasus workflow
```bash
# Local images
./workflow_generator.py --data-source local --image-dir ./field_images --image-size 128 --epochs 10 --output workflow.yml

# Kaggle dataset
./workflow_generator.py --data-source kaggle --kaggle-dataset emmarex/plantdisease --image-size 128 --epochs 10 --output workflow.yml

# Edge-to-cloud DPU mode
./workflow_generator.py --data-source local --image-dir ./field_images --enable-dpu --edge-site edgepool --cloud-site cloudpool --output workflow.yml
```

### Submit and monitor with Pegasus
```bash
pegasus-plan --submit -s condorpool -o local workflow.yml
pegasus-status <run_directory>
pegasus-analyzer <run_directory>
```

### Build Docker container
```bash
cd Docker
docker buildx build --platform linux/amd64,linux/arm64 -f CropHealth_Dockerfile -t kthare10/crophealth:latest --push .
```

## Image Naming Convention

The `Crop___Disease` folder naming convention (from PlantVillage) is how image categories are identified throughout the pipeline. The catalog CSV maps each image path to its crop/disease/treatment metadata. The `DISEASE_INFO` dict in `fetch_crop_images.py` is the authoritative mapping.

## Kaggle Setup

Set credentials before running Kaggle-sourced steps:
```bash
export KAGGLE_USERNAME="your_username"
export KAGGLE_KEY="your_api_key"
```
Also accept dataset terms at `https://www.kaggle.com/datasets/emmarex/plantdisease` before first download.

## Key Files

| File | Purpose |
|------|---------|
| `workflow_generator.py` | Pegasus DAG generator; defines all job dependencies and resource requirements |
| `fetch_crop_images.py` | Image sourcing; `DISEASE_INFO` dict maps folder names to metadata |
| `crop_catalog.csv` | Sample catalog shipped with repo |
| `Docker/CropHealth_Dockerfile` | Multi-platform container definition |
| `Access-CropHealth-workflow.ipynb` | Notebook for running on ACCESS resources |
