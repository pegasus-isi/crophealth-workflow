# Crop Health Workflow - Pegasus WMS on FABRIC

A Pegasus workflow system for detecting and classifying crop diseases from images using deep learning, with treatment and fertilizer recommendations.

## Overview

This workflow processes crop images through a CNN-based disease detection pipeline, providing actionable insights for farmers including disease identification, severity assessment, and treatment recommendations.

### Workflow Architecture

```
Image Source → Fetch/Catalog Images → Preprocess Images → Train Classifier
                                                               │
                        ┌──────────────────────────────────────┘
                        ↓
               Classify Diseases → Evaluate Accuracy → Generate Report → HTML + Charts + JSON
```

### DAG Visualization

The following diagram shows the workflow DAG:

![Crop Health Workflow DAG](images/workflow.png)

### Workflow Modes

`--mode` selects which part of the pipeline to generate. `train` and `inference`
follow the ACCESS Pegasus tutorials
([05-Tutorial-ML-Training](https://github.com/pegasus-isi/ACCESS-Pegasus-Examples/tree/main/05-Tutorial-ML-Training),
[06-Tutorial-ML-Inference](https://github.com/pegasus-isi/ACCESS-Pegasus-Examples/tree/main/06-Tutorial-ML-Inference)):
train a model once, then reuse it over and over on new images.

| Mode | DAG |
|------|-----|
| `full` (default) | fetch → preprocess → train → classify → evaluate → report |
| `train` | [fetch →] preprocess → train |
| `inference` | classify (one job per image, in parallel) → merge → report |

```
inference:   classify_00 ─┐
             classify_01 ─┼─→ merge_predictions → generate_report
             classify_NN ─┘
```

Inference mode has no ground truth for the new images, so it skips
`evaluate_accuracy` and the report has no confusion matrix.

## Features

- **Multi-crop support**: Apple, Corn, Grape, Potato, Tomato, and more
- **38+ disease types**: Comprehensive disease detection across crops
- **CNN classification**: PyTorch-based deep learning with transfer learning
- **Severity scoring**: Critical, High, Moderate, Low, None
- **Treatment recommendations**: Actionable guidance for each disease
- **Fertilizer guidance**: Crop-specific fertilizer recommendations
- **Accuracy evaluation**: Per-class precision, recall, F1, and confusion matrix
- **Rich visualizations**: Distribution charts, confidence histograms, confusion matrix heatmap
- **HTML reports**: Professional reports with all findings including accuracy metrics
- **Train once, infer many times**: separate `train` and `inference` modes; inference fans out one job per image

## Running on ACCESS

The easiest way to run this workflow is using the provided Jupyter notebook on an ACCESS resource with Pegasus and HTCondor pre-configured:

**Notebook**: [`Access-CropHealth-workflow.ipynb`](Access-CropHealth-workflow.ipynb)

The notebook walks through the complete workflow: configuring parameters, generating the Pegasus DAG, submitting to HTCondor, monitoring execution, and examining results with inline visualizations.

## Running on FABRIC

The workflow can also be run on the [FABRIC testbed](https://fabric-testbed.net/) by deploying a distributed Pegasus/HTCondor cluster across FABRIC sites.

### Deploy a Pegasus/HTCondor Cluster

You can provision a cluster using either of the following notebooks:

| Option | Link | Description |
|--------|------|-------------|
| FABRIC Artifact (Recommended) | [Pegasus-FABRIC Artifact](https://artifacts.fabric-testbed.net/artifacts/53da4088-a175-4f0c-9e25-a4a371032a39) | Pre-configured notebook from the FABRIC Artifacts repository |
| Jupyter Examples | [pegasus-fabric.ipynb](https://github.com/fabric-testbed/jupyter-examples/blob/f7be0c75f22544c72d7b3e3fa42bbdfd9d8bb841/fabric_examples/complex_recipes/pegasus/pegasus-fabric.ipynb) | Notebook from the official FABRIC Jupyter examples |

Both notebooks provision the following cluster architecture:

- **Submit Node** -- Central Manager running HTCondor scheduler and Pegasus WMS
- **Worker Nodes** -- Distributed execution points across multiple FABRIC sites
- **FABNetv4 Networking** -- Private L3 network connecting all nodes

### Setup Steps

1. Log into the [FABRIC JupyterHub](https://jupyter.fabric-testbed.net/)
2. Upload or clone one of the Pegasus-FABRIC notebooks above
3. Configure your desired sites and node specifications
4. Run the notebook to provision the cluster
5. Clone this repository on the submit node
6. Run the workflow using the CLI (below) or the [Access notebook](Access-CropHealth-workflow.ipynb)

### Run the Workflow

SSH to the submit node and run:

```bash
cd crophealth-workflow

# Build the container first (see Quick Start step 1 for details)
apptainer build Apptainer/CropHealth_Container.sif \
    Apptainer/CropHealth_Container.def

# Generate workflow
./workflow_generator.py \
    --data-source kaggle \
    --kaggle-dataset emmarex/plantdisease \
    --image-size 128 \
    --epochs 10 \
    -e condorpool \
    --output workflow.yml

# Plan and submit to a plain HTCondor pool (the generator never submits)
pegasus-plan --dir submit -s condorpool -o local --output-dir "$PWD/output" --submit workflow.yml

# Monitor
pegasus-status <run_directory>
```

## Prerequisites

### Software Requirements

- Python 3.9+
- Pegasus WMS v5.0+
- HTCondor v10.2+
- Apptainer (on the submit host to build, and on the worker nodes to run)

### Python Dependencies

```bash
pip install -r requirements.txt
```

### Kaggle API Setup (for PlantVillage dataset)

To download the PlantVillage dataset from Kaggle, you need to set up the Kaggle API:

**Step 1: Create a Kaggle Account**

1. Go to [https://www.kaggle.com/](https://www.kaggle.com/)
2. Click "Register" and create a free account
3. Verify your email address

**Step 2: Generate API Token**

1. Log in to Kaggle
2. Click on your profile icon (top right) → "Settings"
3. Scroll down to the "API" section
4. Click "Create New Token"
5. Copy the token value shown

**Step 3: Configure the API Token**

```bash
# Install the Kaggle package (preferred: kagglehub)
pip install kagglehub

# Or install the legacy client
pip install kaggle

export KAGGLE_USERNAME="your_kaggle_username"
export KAGGLE_KEY="your_kaggle_api_key"
```

**Step 4: Accept Dataset Rules**

1. Go to [https://www.kaggle.com/datasets/emmarex/plantdisease](https://www.kaggle.com/datasets/emmarex/plantdisease)
2. Click "Download" to accept the dataset's terms of use
3. You can cancel the download - you just need to accept the terms

Now you can use the Kaggle data source:

```bash
./fetch_crop_images.py --source kaggle --dataset emmarex/plantdisease \
    --output crop_catalog.csv
```

## Directory Structure

```
crophealth-workflow/
├── workflow_generator.py          # Workflow generator (full / train / inference modes)
├── Access-CropHealth-workflow.ipynb # Notebook driving the same generator code
├── fetch_crop_images.py           # Image fetcher/catalog creator
├── example_usage.sh               # Example usage script
├── bin/
│   ├── preprocess_images.py       # Image preprocessing and augmentation
│   ├── train_classifier.py        # CNN model training
│   ├── classify_disease.py        # Disease inference
│   ├── evaluate_accuracy.py       # Accuracy evaluation against ground truth
│   ├── merge_predictions.py       # Merge per-image predictions (inference mode)
│   └── generate_report.py         # Report generation
├── Apptainer/
│   └── CropHealth_Container.def   # Container definition (built to a .sif)
├── Docker/
│   └── CropHealth_Dockerfile      # Legacy Dockerfile, kept as a fallback
├── models/                        # Trained models
├── output/                        # Workflow outputs
└── README.md
```

## Quick Start

### 1. Build the Container

Do this first — the generator in step 4 expects the `.sif` to exist.

```bash
# Run from the workflow root. No registry push needed: Pegasus stages the .sif
# like any other input file.
apptainer build Apptainer/CropHealth_Container.sif \
    Apptainer/CropHealth_Container.def

# Verify
apptainer exec Apptainer/CropHealth_Container.sif \
    python -c "import torch, kagglehub, sklearn; print('ok')"
apptainer exec Apptainer/CropHealth_Container.sif which curl wget
```

`workflow_generator.py` looks for `Apptainer/CropHealth_Container.sif` by default
(override with `--container-sif`).

Apptainer cannot build on macOS, and a `.sif` is single-architecture — build on a
Linux host matching your worker nodes (an aarch64 pool needs its own `.sif` built
on an aarch64 host). See [`APPTAINER.md`](APPTAINER.md). The legacy
`Docker/CropHealth_Dockerfile` is kept as a fallback.

<details>
<summary>Optional: publish the image to ghcr.io</summary>

Useful for sharing one build across a team or citing an immutable artifact. Needs a
GitHub token with `write:packages`.

```bash
echo "$GHCR_TOKEN" | apptainer registry login --username <github-user> \
    --password-stdin oras://ghcr.io

TAG=$(git rev-parse --short HEAD)
apptainer push Apptainer/CropHealth_Container.sif \
    oras://ghcr.io/pegasus-isi/crophealth-workflow:$TAG

# On the submit host, pull back to the path the generator expects
apptainer pull Apptainer/CropHealth_Container.sif \
    oras://ghcr.io/pegasus-isi/crophealth-workflow:$TAG
```

Do **not** put the `oras://` URL in the transformation catalog — Pegasus supports
`docker://`, `shub://`, `library://`, `shifter://` and `file://`, not `oras://`.
Treat ghcr.io as a distribution channel and keep staging the local `.sif`. Details in
[`APPTAINER.md`](APPTAINER.md).

</details>

### 2. Prepare Your Images

Organize images in disease categories:

```
field_images/
├── Tomato___Early_blight/
│   ├── image1.jpg
│   └── image2.jpg
├── Tomato___healthy/
│   └── image3.jpg
├── Potato___Late_blight/
│   └── image4.jpg
```

### 3. Create Image Catalog

```bash
cd crophealth-workflow

# From local images
./fetch_crop_images.py --source local --input-dir ./field_images \
    --output crop_catalog.csv

# Or use sample data for testing
./fetch_crop_images.py --source sample --output crop_catalog.csv

# Or download from Kaggle (requires API key)
./fetch_crop_images.py --source kaggle --dataset emmarex/plantdisease \
    --output crop_catalog.csv
```

### 4. Generate Workflow

#### Full Mode
##### Using pre downloaded local data
```bash
./workflow_generator.py \
    --data-source local \
    --image-dir ./field_images \
    --image-size 128  \
    --epochs 10  \
    --batch-size 16 \
    --output workflow.yml
```
##### Download data from Kaggle
```bash
./workflow_generator.py  \
    --data-source kaggle  \
    --kaggle-dataset emmarex/plantdisease \
    --image-size 128  \
    --epochs 10  \
    --batch-size 16 \
    --output workflow.yml
```

#### Train Mode

Preprocess and train only. `--data-source remote` uses the catalog and image
archive hosted at `--base-url` (the ACCESS tutorial dataset) instead of a fetch
job; `--gpu` requests a GPU for the training job.

```bash
./workflow_generator.py \
    --mode train \
    --data-source remote \
    --gpu \
    --output workflow.yml
```

#### Inference Mode

Classify images with an already trained model, one job per image. With no
options it uses the hosted model and the 10 hosted sample images
(`<base-url>/inference/00.jpg` … `09.jpg`):

```bash
./workflow_generator.py --mode inference --output workflow.yml
```

To use the model from a training run and your own images (files, directories or
URLs):

```bash
./workflow_generator.py \
    --mode inference \
    --model-dir ./output \
    --inference-images ./new_field_images \
    --output workflow.yml
```

### 5. Choose Where It Runs

Where jobs run depends on your resource provider and allocation, so it lives in
a site catalog you choose, never in the generator, which writes no site
catalog. Jobs run on a site named `compute` (`-e`, the default), the one site
every centrally hosted catalog
([pegasushub/pegasus-site-catalogs](https://github.com/pegasushub/pegasus-site-catalogs/tree/main/conf))
defines; `pegasus-plan` downloads the catalog from the branch matching its
Pegasus version.

**Hosted catalog, per workflow** (e.g. ACCESS Pegasus, Unity):

```bash
./workflow_generator.py -s access-pegasus.yml
pegasus-plan --dir submit -s compute -o local --output-dir "$PWD/output" --submit workflow.yml
```

**Hosted catalog, once per user**: the ACCESS setup notebook (`00-Setup`) names
the catalog and your allocation in `~/.pegasusrc`; elsewhere add them yourself,
then generate with no `-s`:

```bash
cat >> ~/.pegasusrc <<'EOF'
pegasus.catalog.site.repo.file = unity.yml
env.RESOURCE_USERNAME = jdoe
env.RESOURCE_PROJECT = my_lab
EOF
./workflow_generator.py --gpu       # train_classifier gets the catalogs' "gpu" tag
```

**Plain HTCondor pool with no site catalog** (e.g. a FABRIC slice): Pegasus has
no built-in `compute`, but it provides a default `condorpool` site, so:

```bash
./workflow_generator.py -e condorpool
pegasus-plan --dir submit -s condorpool -o local --output-dir "$PWD/output" --submit workflow.yml
```

Outputs land in `./output/`: the printed plan command passes `--output-dir`
(otherwise Pegasus's built-in `local` site would use `./wf-output/`).
`Access-CropHealth-workflow.ipynb` runs the same generator code (`build_parser()`
and `generate()`), writes a local HTCondor `compute` site with
`create_sites_catalog()` when no hosted catalog is set (outputs in
`./output/`), and submits from an explicit cell.

| Option | Default | Meaning |
|---|---|---|
| `-s, --hosted-site-catalog` | (none; `~/.pegasusrc` if set) | Hosted catalog to plan against, e.g. `access-pegasus.yml`, `unity.yml`; written to `pegasus.properties`. |
| `-e, --execution-site-name` | `compute` | Execution site name; `condorpool` on a plain HTCondor pool with no site catalog. |

Notes:

- **Slurm submission** goes through HTCondor's glite/BLAHP, so plan on the
  cluster's login node with HTCondor and Pegasus installed.
- **GPU jobs** (`--gpu`, `--gpu-inference`) request one GPU and carry the
  Pegasus tag `gpu`, which hosted catalogs map to their GPU partition. Tags
  need Pegasus 6.0 (or 5.1.3dev) at plan time.
- **Runtime.** Training states a 6 h budget (`TRAIN_CLASSIFIER_RUNTIME`),
  since it can outlast a hosted batch catalog's default (2 h on Unity); every
  other step uses the catalog's default.

### 6. Plan and Submit

The generator writes the workflow and catalogs and prints this command; it
never submits by itself. Use the `-e` value you generated with as `-s`:

```bash
pegasus-plan --dir submit -s compute -o local --output-dir "$PWD/output" --submit workflow.yml

# Monitor
pegasus-status <run_directory>
```

### 7. View Results

```
output/
├── crop_catalog.csv              # Image catalog
├── train_data.npz                # Preprocessed training data
├── val_data.npz                  # Preprocessed validation data
├── disease_classifier.pt         # Trained model
├── predictions.json              # Disease predictions
├── accuracy_results.json         # Accuracy metrics and confusion matrix
├── report/
│   ├── report.html               # HTML report with accuracy metrics
│   ├── disease_distribution.png  # Disease distribution chart
│   ├── severity_distribution.png # Severity chart
│   ├── crop_health_summary.png   # Crop-wise summary
│   └── confusion_matrix.png      # Confusion matrix heatmap
```

## Helper Scripts

For quick local testing, use the provided scripts:

- `example_usage.sh` shows a lightweight Kaggle-based walkthrough.
- `run_manual.sh` runs the full manual pipeline end-to-end using Kaggle.

Note: You must accept the dataset terms at https://www.kaggle.com/datasets/emmarex/plantdisease before downloading.

## Configuration Options

### Workflow Generator Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--mode` | `full`, `train` or `inference` | full |
| `--data-source` | Training image source (local, kaggle, sample, remote) | sample |
| `--base-url` | Hosted inputs for `remote` data and `inference` mode | https://download.pegasus.isi.edu/tutorial/crophealth |
| `--image-dir` | Local image directory | - |
| `--kaggle-dataset` | Kaggle dataset name | emmarex/plantdisease |
| `--image-size` | Target image size | 224 |
| `--train-split` | Training fraction | 0.8 |
| `--epochs` | Training epochs | 20 |
| `--batch-size` | Training batch size | 32 |
| `--gpu` | Request a GPU for `train_classifier` | False |
| `-s, --hosted-site-catalog` | Hosted site catalog, e.g. `access-pegasus.yml`; see [Choose Where It Runs](#5-choose-where-it-runs) | none (`~/.pegasusrc`) |
| `-e, --execution-site-name` | Execution site; `condorpool` on a plain HTCondor pool with no site catalog | compute |
| `--container-sif` | Apptainer image | Apptainer/CropHealth_Container.sif |
| `-o, --output` | Output workflow file | workflow.yml |

### Inference Mode Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--model-dir` | Directory with `disease_classifier.pt` and `training_info.json` | model at `--base-url` |
| `--inference-images` | Images to classify: files, directories or URLs | hosted samples |
| `--num-remote-images` | Hosted samples to use when `--inference-images` is not given | 10 |
| `--gpu-inference` | Request a GPU for each classify job | False |

## Supported Diseases

### Apple
| Disease | Severity | Treatment |
|---------|----------|-----------|
| Apple Scab | Moderate | Fungicide (captan, mancozeb) |
| Black Rot | High | Remove infected fruit/branches |
| Cedar Apple Rust | Moderate | Remove cedars, apply fungicide |

### Corn
| Disease | Severity | Treatment |
|---------|----------|-----------|
| Gray Leaf Spot | Moderate | Resistant varieties, fungicide |
| Common Rust | Moderate | Fungicide, resistant hybrids |
| Northern Leaf Blight | Moderate | Crop rotation, fungicide |

### Grape
| Disease | Severity | Treatment |
|---------|----------|-----------|
| Black Rot | High | Mancozeb/myclobutanil fungicide |
| Esca | High | Prune infected wood (no cure) |
| Leaf Blight | Moderate | Fungicide application |

### Potato
| Disease | Severity | Treatment |
|---------|----------|-----------|
| Early Blight | Moderate | Chlorothalonil/copper fungicide |
| Late Blight | Critical | Destroy plants, immediate fungicide |

### Tomato
| Disease | Severity | Treatment |
|---------|----------|-----------|
| Bacterial Spot | Moderate | Copper-based bactericide |
| Early Blight | Moderate | Fungicide, remove lower leaves |
| Late Blight | Critical | Destroy infected plants |
| Leaf Mold | Moderate | Improve ventilation, fungicide |
| Septoria Leaf Spot | Moderate | Remove leaves, fungicide |
| Yellow Leaf Curl Virus | High | Remove plants, control whiteflies |
| Mosaic Virus | High | Remove plants, sanitize tools |

## Output Files

### Predictions JSON

```json
{
  "predictions": [
    {
      "filename": "tomato_leaf_001.jpg",
      "crop": "Tomato",
      "disease": "Early Blight",
      "is_healthy": false,
      "confidence": 0.9523,
      "treatment": {
        "severity": "moderate",
        "action": "Apply copper-based fungicide",
        "prevention": "Mulch around plants, avoid overhead watering",
        "fertilizer": "Calcium and potassium to strengthen plants"
      }
    }
  ],
  "summary": {
    "total": 100,
    "healthy": 25,
    "diseased": 75,
    "disease_breakdown": {
      "Early Blight": 30,
      "Late Blight": 15,
      "Healthy": 25
    }
  },
  "critical_alerts": [
    {
      "image": "potato_leaf_042.jpg",
      "disease": "Late Blight",
      "action": "Destroy infected plants immediately"
    }
  ]
}
```

### Accuracy Results JSON

```json
{
  "evaluated_at": "2026-02-04T12:00:00",
  "overall_accuracy": 0.95,
  "total_evaluated": 4627,
  "correct": 4400,
  "incorrect": 227,
  "unmatched": 0,
  "per_class": {
    "Pepper__bell___Bacterial_spot": {
      "total": 997, "correct": 980, "accuracy": 0.983,
      "precision": 0.98, "recall": 0.983, "f1": 0.981
    }
  },
  "confusion_matrix": {
    "labels": ["Pepper__bell___Bacterial_spot", "..."],
    "matrix": [[980, 5], [3, 500]]
  }
}
```

### HTML Report

Interactive HTML report with:
- Summary statistics
- Critical alerts
- Accuracy metrics (overall and per-class precision/recall/F1)
- Confusion matrix heatmap
- Disease distribution charts
- Severity breakdown
- Treatment recommendations
- Detailed results table

## Advanced Usage

### Training with GPU

In a workflow, pass `--gpu` to the generator: the `train_classifier` job then
requests a GPU and uses CUDA. To train by hand:

```bash
./bin/train_classifier.py \
    --input-dir ./processed \
    --output-dir ./models \
    --epochs 50 \
    --device cuda
```

### Custom Container

See [Quick Start step 1](#1-build-the-container) for the build command, and
[`APPTAINER.md`](APPTAINER.md) for the definition-file reference and the
ghcr.io publishing recipe. Point the generator at a different image with
`--container-sif /path/to/your.sif`.

### Using Pre-trained Model

As a workflow, use [Inference Mode](#inference-mode). By hand:

```bash
./bin/classify_disease.py \
    --model-dir ./pretrained_models \
    --input ./new_field_images \
    --output predictions.json
```

## Troubleshooting

### Common Issues

**1. No images found**
- Verify image directory structure matches expected format
- Check file extensions (.jpg, .jpeg, .png)
- Ensure category folders follow `Crop___Disease` naming

**2. Training fails with memory error**
- Reduce batch size (`--batch-size 16`)
- Reduce image size (`--image-size 128`)
- Use sklearn fallback (`--use-sklearn`)

**3. Low classification accuracy**
- Increase training epochs
- Ensure sufficient training data (100+ images per class)
- Try data augmentation

**4. Inference `merge` job fails with "No successful predictions"**
- Every classify job failed to read its image; check the per-image
  `*_predictions.json` files for the `error` field

### Debugging

```bash
# View Pegasus logs
pegasus-analyzer <run_directory>

# Check job logs
cat <run_directory>/work/<job_name>/*.err
cat <run_directory>/work/<job_name>/*.out
```

## Related Resources

- [PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Pegasus WMS Documentation](https://pegasus.isi.edu/documentation/)
- [FABRIC Testbed](https://portal.fabric-testbed.net/)

## Citation

```
@misc{crophealth-workflow,
  title={Crop Disease Detection Workflow using Pegasus WMS},
  year={2025},
  publisher={GitHub},
  url={https://github.com/pegasus-isi/crophealth-workflow}
}
```

## License

This workflow is released under the same license as the parent repository.

## Contributing

Contributions welcome! Please submit issues or pull requests for:
- Additional crop/disease types
- Improved ML models
- New data sources
- Performance improvements

---
## Authors
Komal Thareja (kthare10@renci.org)

Built with the assistance of Claude.
