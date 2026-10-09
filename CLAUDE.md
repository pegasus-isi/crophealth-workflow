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

- **`custom_sites.py`** — Site-catalog logic (`ensure_sites_yml`), shared with airquality-workflow; imported by the generator and runnable standalone.

**Sites**: the workflow is site-agnostic. Transformations are registered on `local` with cores/memory/`runtime` (`TOOL_RUNTIME`); `train_classifier` carries the tag `train`. `sites.yml` precedence: an existing entry, then a hosted catalog named with `-s FILE` (written to `pegasus.properties`) or in `~/.pegasusrc` (site `compute`), then a default HTCondor site; `-e` always defaults to `compute`; `local` is always ensured. Over a hosted catalog, `--site-style` writes an overlay only for a site the catalog defines, a full entry otherwise. `--site-style slurm --queue --project` targets batch clusters; there `--shared-filesystem auto` turns on bypass staging and the container binds the workflow dir. With `pegasus-version` available, a `rhel_8` `pegasus::worker` package is staged into the Debian 13 (trixie) container.

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
# Slurm instead of the default HTCondor site
./workflow_generator.py -e compute --site-style slurm --queue <partition> --project <account> --train-profile pegasus:queue=<gpu-partition>
```

### Submit and monitor with Pegasus
```bash
pegasus-plan --submit -s compute -o local workflow.yml   # use the site you generated for
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
| `custom_sites.py` | Writes `sites.yml` (HTCondor / Slurm / hosted-catalog overlay) |
| `fetch_crop_images.py` | Image sourcing; `DISEASE_INFO` dict maps folder names to metadata |
| `crop_catalog.csv` | Sample catalog shipped with repo |
| `Docker/CropHealth_Dockerfile` | Multi-platform container definition |
| `Access-CropHealth-workflow.ipynb` | Notebook for running on ACCESS resources |
