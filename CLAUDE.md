# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a **Pegasus WMS workflow** for detecting and classifying crop diseases from images using CNN-based deep learning. It targets execution on ACCESS (XSEDE) and FABRIC testbed infrastructure via HTCondor.

## Architecture

`workflow_generator.py --mode` selects the DAG (`train`/`inference` mirror the ACCESS Pegasus tutorials 05-Tutorial-ML-Training and 06-Tutorial-ML-Inference):

```
full (default):  fetch_images → preprocess → train → classify → evaluate → report
train:           [fetch_images →] preprocess → train
inference:       classify_<img> (one per image, parallel) → merge → report
```

`--data-source remote` replaces the fetch job with the catalog/archive hosted at `--base-url` (default `https://download.pegasus.isi.edu/tutorial/crophealth`). Inference takes the model from `--model-dir` or `--base-url`, and images from `--inference-images` (files/dirs/URLs) or `<base-url>/inference/NN.jpg`. Inference has no ground truth, so no evaluate job and no confusion matrix.

- **`workflow_generator.py`** — Generates the Pegasus workflow YAML (`workflow.yml`) using `Pegasus.api`. This is the entry point for creating the workflow.
- **`fetch_crop_images.py`** — Fetches/catalogs images from local disk, Kaggle, or sample data; outputs `crop_catalog.csv` + `images.tar.gz`.
- **`bin/preprocess_images.py`** — Resizes and normalizes images; outputs `train_data.npz`, `val_data.npz`, `label_mapping.json`.
- **`bin/train_classifier.py`** — Trains PyTorch CNN (transfer learning); outputs `disease_classifier.pt`.
- **`bin/classify_disease.py`** — Runs inference; outputs `predictions.json`.
- **`bin/evaluate_accuracy.py`** — Computes per-class precision/recall/F1 and confusion matrix; outputs `accuracy_results.json`.
- **`bin/merge_predictions.py`** — Inference mode: merges per-image `*_predictions.json` into `predictions.json`.
- **`bin/generate_report.py`** — Generates HTML report + PNG charts.

- **`Access-CropHealth-workflow.ipynb`** — Notebook: builds the CLI's argv, calls `build_parser()`/`generate(args, sites_catalog=...)` from the generator (no copied workflow code), submits from an explicit cell via `plan_submit()`.

**Sites** (pegasus-isi/pegasus-gromacs pattern): the CLI writes no site catalog and never submits — it prints the `pegasus-plan` command. Jobs run on `-e compute` (default), defined by a hosted catalog named with `-s FILE` (written to `pegasus.properties`) or in `~/.pegasusrc`; `-e condorpool` on a plain HTCondor pool with no catalog. `create_sites_catalog()` (local + HTCondor `compute`) is a placeholder only the notebook uses (`generate(args, sites_catalog=True)`). Transformations are registered on the execution site with cores/memory; `train_classifier` states a 6 h `runtime`. `--gpu`/`--gpu-inference` jobs request a GPU and carry the hosted catalogs' `gpu` tag.

All jobs run inside an Apptainer image, `Apptainer/CropHealth_Container.sif` by default (`--container-sif`), built locally and staged by Pegasus; see `APPTAINER.md`.

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

# Train only, on the hosted tutorial dataset, with a GPU
./workflow_generator.py --mode train --data-source remote --gpu --output workflow.yml

# Inference: hosted model over the 10 hosted sample images
./workflow_generator.py --mode inference --output workflow.yml

# Inference: your trained model over your images
./workflow_generator.py --mode inference --model-dir ./output --inference-images ./new_images --output workflow.yml
# A hosted site catalog (or one in ~/.pegasusrc) / a plain HTCondor pool
./workflow_generator.py -s unity.yml --gpu
./workflow_generator.py -e condorpool
```

### Submit and monitor with Pegasus
```bash
pegasus-plan --dir submit -s compute -o local --output-dir "$PWD/output" --submit workflow.yml   # -s = the -e value
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
| `Access-CropHealth-workflow.ipynb` | Notebook driving the generator's `build_parser()`/`generate()` |
| `fetch_crop_images.py` | Image sourcing; `DISEASE_INFO` dict maps folder names to metadata |
| `crop_catalog.csv` | Sample catalog shipped with repo |
| `Docker/CropHealth_Dockerfile` | Multi-platform container definition |
| `Access-CropHealth-workflow.ipynb` | Notebook for running on ACCESS resources |
