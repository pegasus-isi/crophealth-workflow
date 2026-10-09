#!/usr/bin/env python3

"""
Crop Health Workflow Generator for Pegasus WMS.

Three modes (--mode):

  full (default)  fetch -> preprocess -> train -> classify -> evaluate -> report
  train           [fetch ->] preprocess -> train
  inference       classify (one job per image, in parallel) -> merge -> report

train and inference follow the ACCESS Pegasus tutorials (05-Tutorial-ML-Training
and 06-Tutorial-ML-Inference): a model is trained once, then reused over and
over on new images. With --data-source remote the training inputs come from
--base-url instead of a fetch job; inference takes the model from --model-dir
or --base-url, and the images from --inference-images or --base-url.

Usage:
    # Full pipeline with local images
    ./workflow_generator.py --data-source local --image-dir ./field_images

    # Train on the hosted tutorial dataset, requesting a GPU
    ./workflow_generator.py --mode train --data-source remote --gpu

    # Inference with the hosted model over the 10 hosted sample images
    ./workflow_generator.py --mode inference

    # Inference with a locally trained model over your own images
    ./workflow_generator.py --mode inference --model-dir ./output \
        --inference-images ./new_images/*.jpg

    # A plain HTCondor pool with no site catalog
    ./workflow_generator.py --mode inference -e condorpool

Sites follow pegasus-isi/pegasus-gromacs: jobs run on a site named "compute",
defined by a centrally hosted site catalog (-s FILE, or one in ~/.pegasusrc;
https://github.com/pegasushub/pegasus-site-catalogs). The generator writes no
site catalog and never submits: it prints the pegasus-plan command, and the
notebook (Access-CropHealth-workflow.ipynb) submits from an explicit cell.
"""

import argparse
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path

from Pegasus.api import *


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Hosted inputs used by the ACCESS Pegasus tutorials: crop_catalog.csv,
# images.tar.gz, disease_classifier.pt, training_info.json, inference/NN.jpg
DEFAULT_BASE_URL = "https://download.pegasus.isi.edu/tutorial/crophealth"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}


# Training can outlast a hosted batch catalog's default wall-clock (2 h on
# Unity), so it states its own budget, in seconds.
TRAIN_CLASSIFIER_RUNTIME = 6 * 3600

# Tag the hosted site catalogs map to their GPU partition (x-tags "gpu");
# carried by jobs that request a GPU (--gpu, --gpu-inference).
GPU_TAG = "gpu"


class CropHealthWorkflow:
    """Generate Pegasus workflow for crop disease detection."""

    wf = None
    sc = None
    tc = None
    rc = None
    props = None

    dagfile = None
    wf_dir = None
    shared_scratch_dir = None
    local_storage_dir = None
    wf_name = "crophealth"

    def __init__(self, dagfile="workflow.yml"):
        """Initialize workflow."""
        self.dagfile = dagfile
        self.wf_dir = str(Path(__file__).parent.resolve())
        self.shared_scratch_dir = os.path.join(self.wf_dir, "scratch")
        self.local_storage_dir = os.path.join(self.wf_dir, "output")

    def write(self):
        """Write all catalogs and workflow to files."""
        if self.sc is not None:
            self.sc.write()
        self.props.write()
        self.rc.write()
        self.tc.write()
        self.wf.write(file=self.dagfile)

    # ------------------------------------------------------------------
    # Plan / run / monitor (thin wrappers over the Pegasus API Workflow
    # object, for interactive use e.g. from a Jupyter notebook)
    # ------------------------------------------------------------------
    def plan_submit(self, exec_site_name="compute", raise_errors=False):
        try:
            self.wf.plan(
                dir="submit",
                sites=[exec_site_name],
                output_sites=["local"],
                cleanup="none",
                verbose=1,
                submit=True,
            )
        except PegasusClientError as e:
            print(e)
            if raise_errors:
                raise

    def status(self):
        try:
            self.wf.status(long=True)
        except PegasusClientError as e:
            print(e)

    def wait(self):
        try:
            self.wf.wait()
        except PegasusClientError as e:
            print(e)

    def statistics(self):
        try:
            self.wf.statistics()
        except PegasusClientError as e:
            print(e)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    def create_pegasus_properties(self, hosted_site_catalog=None):
        self.props = Properties()
        self.props["pegasus.transfer.threads"] = "16"
        if hosted_site_catalog:
            # Use one of Pegasus' centrally hosted site catalogs instead of
            # a locally generated one. pegasus-plan downloads and caches the
            # named file from the catalog repository at plan time.
            # https://pegasus.isi.edu/documentation/reference-guide/catalogs.html#centrally-hosted-site-catalogs
            self.props["pegasus.catalog.site.repo.file"] = hosted_site_catalog

    # ------------------------------------------------------------------
    # Site Catalog
    #
    # Not used by the CLI below by default — pegasus-plan resolves the site
    # catalog from a centrally hosted one instead (see -s/--hosted-site-catalog
    # and create_pegasus_properties above). Kept for programmatic/notebook use
    # when a self-contained, locally generated HTCondor site catalog is wanted.
    # ------------------------------------------------------------------
    def create_sites_catalog(self, exec_site_name="compute"):
        self.sc = SiteCatalog()

        local = Site("local").add_directories(
            Directory(
                Directory.SHARED_SCRATCH, self.shared_scratch_dir
            ).add_file_servers(
                FileServer("file://" + self.shared_scratch_dir, Operation.ALL)
            ),
            Directory(
                Directory.LOCAL_STORAGE, self.local_storage_dir
            ).add_file_servers(
                FileServer("file://" + self.local_storage_dir, Operation.ALL)
            ),
        )

        exec_site = (
            Site(exec_site_name)
            .add_condor_profile(universe="vanilla")
            .add_pegasus_profile(style="condor")
        )

        self.sc.add_sites(local, exec_site)

    def create_replica_catalog(self):
        """Create replica catalog for input files."""
        logger.info("Creating replica catalog")
        self.rc = ReplicaCatalog()

    def create_transformation_catalog(
        self,
        exec_site_name="compute",
        container_sif="Apptainer/CropHealth_Container.sif",
        gpu=False,
        large_memory="32 GB",
    ):
        logger.info("Creating transformation catalog")
        self.tc = TransformationCatalog()

        # Container - a local Apptainer .sif built with `apptainer build`.
        # Pegasus stages the file like any other input, so image_site is the
        # site where the .sif physically lives (the submit host = "local").
        sif_path = (
            container_sif
            if os.path.isabs(container_sif)
            else os.path.join(self.wf_dir, container_sif)
        )
        if not os.path.exists(sif_path):
            logger.warning(
                "Apptainer image not found at %s — build it first with: "
                "apptainer build %s Apptainer/CropHealth_Container.def",
                sif_path,
                sif_path,
            )
        crophealth_container = Container(
            "crophealth_container",
            container_type=Container.SINGULARITY,
            image="file://" + sif_path,
            image_site="local",
        )

        # Scripts live on the submit host and are staged to the execution site.
        def tool(name, script, memory):
            return Transformation(
                name,
                site=exec_site_name,
                pfn=os.path.join(self.wf_dir, script),
                is_stageable=True,
                container=crophealth_container,
            ).add_pegasus_profile(cores=1, memory=memory)

        fetch_crop_images = tool("fetch_crop_images", "fetch_crop_images.py", "4 GB")
        # preprocess and train hold the whole dataset in memory: about 22 GB
        # peak for the 9k-image tutorial set at 224 px, ~7 GB at 128 px
        preprocess_images = tool("preprocess_images", "bin/preprocess_images.py", large_memory)
        train_classifier = tool("train_classifier", "bin/train_classifier.py", large_memory)
        train_classifier.add_pegasus_profile(runtime=TRAIN_CLASSIFIER_RUNTIME)
        # train_classifier.py uses CUDA whenever the job lands on a GPU
        if gpu:
            train_classifier.add_pegasus_profile(gpus="1")
        # classify reads one image at a time; report reads only predictions
        classify_disease = tool("classify_disease", "bin/classify_disease.py", "8 GB")
        evaluate_accuracy = tool("evaluate_accuracy", "bin/evaluate_accuracy.py", "4 GB")
        merge_predictions = tool("merge_predictions", "bin/merge_predictions.py", "2 GB")
        generate_report = tool("generate_report", "bin/generate_report.py", "4 GB")

        self.tc.add_containers(crophealth_container)
        self.tc.add_transformations(
            fetch_crop_images,
            preprocess_images,
            train_classifier,
            classify_disease,
            evaluate_accuracy,
            merge_predictions,
            generate_report,
        )

    def add_input(self, lfn, location):
        """Register an input in the replica catalog: a URL or a local path."""
        if re.match(r"^[a-z]+://", location):
            self.rc.add_replica("remote", lfn, location)
        else:
            self.rc.add_replica("local", lfn, "file://" + os.path.abspath(location))

    def create_workflow(self, args):
        """Create the workflow for the selected --mode."""
        logger.info(f"Creating workflow (mode: {args.mode})")
        self.wf = Workflow(self.wf_name)

        if args.mode == "inference":
            self.add_inference_jobs(args)
            return

        catalog_file, images_archive, train_job, model_checkpoint, training_info = (
            self.add_training_jobs(args)
        )
        if args.mode == "full":
            self.add_evaluation_jobs(
                train_job, catalog_file, images_archive, model_checkpoint, training_info
            )

    def add_training_jobs(self, args):
        """Add [fetch ->] preprocess -> train. Returns the files later jobs need."""
        catalog_file = File("crop_catalog.csv")
        images_archive = File("images.tar.gz")
        train_data = File("train_data.npz")
        val_data = File("val_data.npz")
        label_mapping = File("label_mapping.json")
        preprocessing_info = File("preprocessing_info.json")
        model_checkpoint = File("disease_classifier.pt")
        training_info = File("training_info.json")

        fetch_job = None
        if args.data_source == "remote":
            # Pre-hosted inputs, as in the ACCESS tutorial: no fetch job
            base_url = args.base_url.rstrip("/")
            self.add_input(catalog_file, f"{base_url}/{catalog_file.lfn}")
            self.add_input(images_archive, f"{base_url}/{images_archive.lfn}")
        else:
            fetch_job = self.create_fetch_job(args, catalog_file, images_archive)

        # Preprocess images
        preprocess_job = Job("preprocess_images", _id="preprocess", node_label="preprocess")
        preprocess_job.add_args(
            "--input", catalog_file,
            "--output-dir", ".",
            "--image-size", str(args.image_size),
            "--split", str(args.train_split),
            "--images-archive", images_archive,
        )
        preprocess_job.add_inputs(catalog_file, images_archive)
        preprocess_job.add_outputs(train_data, stage_out=True, register_replica=False)
        preprocess_job.add_outputs(val_data, stage_out=True, register_replica=False)
        preprocess_job.add_outputs(label_mapping, stage_out=True, register_replica=False)
        preprocess_job.add_outputs(preprocessing_info, stage_out=True, register_replica=False)
        preprocess_job.add_pegasus_profile(label="preprocess")

        # Train classifier
        train_job = Job("train_classifier", _id="train", node_label="train")
        train_job.add_args(
            "--input-dir", ".",
            "--output-dir", ".",
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
        )
        train_job.add_inputs(train_data, val_data, label_mapping)
        train_job.add_outputs(model_checkpoint, stage_out=True, register_replica=False)
        train_job.add_outputs(training_info, stage_out=True, register_replica=False)
        train_job.add_pegasus_profile(label="train")
        if args.gpu:
            # Hosted catalogs route the "gpu" tag to their GPU partition.
            # (add_profiles, not add_pegasus_profile(tag=...): the keyword
            # only exists in Pegasus >= 5.1.3dev's Python API.)
            train_job.add_profiles(Namespace.PEGASUS, key="tag", value=GPU_TAG)

        self.wf.add_jobs(preprocess_job, train_job)
        if fetch_job is not None:
            self.wf.add_jobs(fetch_job)
            self.wf.add_dependency(fetch_job, children=[preprocess_job])
        self.wf.add_dependency(preprocess_job, children=[train_job])

        return catalog_file, images_archive, train_job, model_checkpoint, training_info

    def create_fetch_job(self, args, catalog_file, images_archive):
        """Create the job that fetches/catalogs images (local, kaggle, sample)."""
        fetch_job = Job("fetch_crop_images", _id="fetch_images", node_label="fetch_images")
        fetch_job.add_args(
            "--source", args.data_source,
            "--output", catalog_file,
            "--output-dir", "./images",
            "--archive-output", images_archive,
        )
        if args.data_source == "local" and args.image_dir:
            fetch_job.add_args("--input-dir", args.image_dir)
        elif args.data_source == "kaggle":
            fetch_job.add_args("--dataset", args.kaggle_dataset)
        # Credentials reach a job only through add_env: HTCondor runs jobs in a
        # clean environment, so exporting KAGGLE_KEY in the submit shell never
        # reaches the job or its container. Capture it here, at generation
        # time.
        #
        # Not defaulted to "": an empty key is indistinguishable from a real
        # one until the job is on a worker node, where the failure reads as a
        # Kaggle API error rather than "you did not set a credential". Only the
        # kaggle source needs them, so a missing key is fatal there and
        # irrelevant otherwise.
        if args.data_source == "kaggle":
            missing = [
                name for name in ("KAGGLE_USERNAME", "KAGGLE_KEY")
                if not os.environ.get(name)
            ]
            if missing:
                logger.error(
                    "--data-source kaggle needs %s in the environment of "
                    "whatever runs this generator. In PegasusAI Studio set "
                    "them under Settings > Pegasus Options > Workflow secrets; "
                    "from a shell, export them before generating.",
                    " and ".join(missing),
                )
                sys.exit(1)
            fetch_job.add_env(KAGGLE_USERNAME=os.environ["KAGGLE_USERNAME"])
            fetch_job.add_env(KAGGLE_KEY=os.environ["KAGGLE_KEY"])
        fetch_job.add_outputs(catalog_file, stage_out=True, register_replica=False)
        fetch_job.add_outputs(images_archive, stage_out=False, register_replica=False)
        fetch_job.add_pegasus_profile(label="fetch")
        return fetch_job

    def add_evaluation_jobs(
        self, train_job, catalog_file, images_archive, model_checkpoint, training_info
    ):
        """Full mode: classify the whole archive -> evaluate -> report."""
        predictions_file = File("predictions.json")
        accuracy_results = File("accuracy_results.json")

        # Classify diseases
        classify_job = Job("classify_disease", _id="classify", node_label="classify")
        classify_job.add_args(
            "--model-dir", ".",
            "--input", "./images",
            "--output", predictions_file,
            "--images-archive", images_archive,
        )
        classify_job.add_inputs(model_checkpoint, training_info, images_archive)
        classify_job.add_outputs(predictions_file, stage_out=True, register_replica=False)
        classify_job.add_pegasus_profile(label="classify")

        # Evaluate accuracy
        evaluate_job = Job("evaluate_accuracy", _id="evaluate", node_label="evaluate")
        evaluate_job.add_args(
            "--predictions", predictions_file,
            "--catalog", catalog_file,
            "--output", accuracy_results,
        )
        evaluate_job.add_inputs(predictions_file, catalog_file)
        evaluate_job.add_outputs(accuracy_results, stage_out=True, register_replica=False)
        evaluate_job.add_pegasus_profile(label="evaluate")

        report_job = self.create_report_job(predictions_file, accuracy_results)

        self.wf.add_jobs(classify_job, evaluate_job, report_job)
        self.wf.add_dependency(train_job, children=[classify_job])
        self.wf.add_dependency(classify_job, children=[evaluate_job])
        self.wf.add_dependency(evaluate_job, children=[report_job])

    def add_inference_jobs(self, args):
        """Inference mode: one classify job per image -> merge -> report."""
        model_checkpoint = File("disease_classifier.pt")
        training_info = File("training_info.json")
        base_url = args.base_url.rstrip("/")

        # A model trained earlier (e.g. by --mode train), or the hosted one
        model_location = args.model_dir or base_url
        for f in (model_checkpoint, training_info):
            self.add_input(f, f"{model_location.rstrip('/')}/{f.lfn}")

        images = resolve_inference_images(args)
        logger.info(f"Inference over {len(images)} images")

        predictions_file = File("predictions.json")
        merge_job = Job("merge_predictions", _id="merge", node_label="merge")
        merge_job.add_args("--output", predictions_file)
        merge_job.add_outputs(predictions_file, stage_out=True, register_replica=False)
        merge_job.add_pegasus_profile(label="merge")

        used_ids = set()
        for location in images:
            name = location.rstrip("/").rsplit("/", 1)[-1]
            # Job IDs and prediction LFNs come from the sanitized stem, which
            # can collide (a.jpg / a.png, "a b.jpg" / a_b.jpg): suffix those
            stem = re.sub(r"[^A-Za-z0-9_-]", "_", Path(name).stem)
            image_id, n = stem, 1
            while image_id in used_ids:
                n += 1
                image_id = f"{stem}_{n}"
            used_ids.add(image_id)

            image_file = File(name)
            self.add_input(image_file, location)
            image_predictions = File(f"{image_id}_predictions.json")

            classify_job = Job(
                "classify_disease", _id=f"classify_{image_id}", node_label=f"classify_{image_id}"
            )
            classify_job.add_args(
                "--model-dir", ".",
                "--input", image_file,
                "--output", image_predictions,
            )
            classify_job.add_inputs(model_checkpoint, training_info, image_file)
            classify_job.add_outputs(image_predictions, stage_out=True, register_replica=False)
            # One image needs far less than the whole-archive classify job
            classify_job.add_pegasus_profile(label="classify", cores="1", memory="4 GB")
            if args.gpu_inference:
                classify_job.add_pegasus_profile(gpus="1")
                classify_job.add_profiles(Namespace.PEGASUS, key="tag", value=GPU_TAG)

            merge_job.add_args(image_predictions)
            merge_job.add_inputs(image_predictions)
            self.wf.add_jobs(classify_job)
            self.wf.add_dependency(classify_job, children=[merge_job])

        report_job = self.create_report_job(predictions_file)
        self.wf.add_jobs(merge_job, report_job)
        self.wf.add_dependency(merge_job, children=[report_job])

    def create_report_job(self, predictions_file, accuracy_results=None):
        """Create the report job; the confusion matrix needs accuracy results."""
        outputs = [
            "report.html",
            "report_summary.json",
            "disease_distribution.png",
            "severity_distribution.png",
            "crop_health_summary.png",
            "confidence_histogram.png",
        ]

        report_job = Job("generate_report", _id="report", node_label="report")
        report_job.add_args(
            "--predictions", predictions_file,
            "--output-dir", ".",
            "--format", "all",
        )
        report_job.add_inputs(predictions_file)
        if accuracy_results is not None:
            report_job.add_args("--accuracy", accuracy_results)
            report_job.add_inputs(accuracy_results)
            outputs.append("confusion_matrix.png")
        for lfn in outputs:
            report_job.add_outputs(File(lfn), stage_out=True, register_replica=False)
        report_job.add_pegasus_profile(label="report")
        return report_job


def resolve_inference_images(args):
    """Return image locations (URLs or local paths) for inference mode."""
    if not args.inference_images:
        base_url = args.base_url.rstrip("/")
        return [
            f"{base_url}/inference/{i:02d}.jpg" for i in range(args.num_remote_images)
        ]

    images = []
    for entry in args.inference_images:
        if os.path.isdir(entry):
            images.extend(
                str(p) for p in sorted(Path(entry).rglob("*"))
                if p.suffix.lower() in IMAGE_EXTENSIONS
            )
        elif re.match(r"^[a-z]+://", entry) or os.path.isfile(entry):
            images.append(entry)
        else:
            raise ValueError(f"Inference image not found: {entry}")
    if not images:
        raise ValueError("No images found in --inference-images")

    # Each image becomes an LFN, so names must be unique across the inputs
    names = [i.rstrip("/").rsplit("/", 1)[-1] for i in images]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise ValueError(f"Duplicate image file names: {', '.join(duplicates)}")
    return images


def build_parser():
    """Build the command-line parser (also used by the ACCESS notebook)."""
    parser = argparse.ArgumentParser(
        description="Generate Pegasus workflow for crop disease detection"
    )

    # Mode
    parser.add_argument(
        "--mode",
        type=str,
        choices=["full", "train", "inference"],
        default="full",
        help="full: train and evaluate end to end; train: preprocess + train "
             "only; inference: classify images with an already trained model"
    )

    # Data source
    parser.add_argument(
        "--data-source",
        type=str,
        choices=["local", "kaggle", "sample", "remote"],
        default="sample",
        help="Source of training images (full/train modes). remote uses the "
             "crop_catalog.csv and images.tar.gz hosted at --base-url"
    )

    parser.add_argument(
        "--base-url",
        type=str,
        default=DEFAULT_BASE_URL,
        help="Where hosted inputs live: catalog and image archive for "
             "--data-source remote; model and inference/NN.jpg images for "
             f"--mode inference (default: {DEFAULT_BASE_URL})"
    )

    parser.add_argument(
        "--image-dir",
        type=str,
        help="Directory with local images (for local source)"
    )

    parser.add_argument(
        "--kaggle-dataset",
        type=str,
        default="emmarex/plantdisease",
        help="Kaggle dataset name (for kaggle source)"
    )

    # Model training parameters
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Target image size for training"
    )

    parser.add_argument(
        "--train-split",
        type=float,
        default=0.8,
        help="Training set fraction"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training batch size"
    )

    # Inference
    parser.add_argument(
        "--model-dir",
        type=str,
        help="Directory with disease_classifier.pt and training_info.json from "
             "a training run (inference mode; default: the model at --base-url)"
    )

    parser.add_argument(
        "--inference-images",
        nargs="+",
        help="Images to classify in inference mode: local files, directories "
             "or URLs (default: the hosted samples at --base-url/inference/)"
    )

    parser.add_argument(
        "--num-remote-images",
        type=int,
        default=10,
        help="How many hosted sample images (00.jpg, 01.jpg, ...) to classify "
             "when --inference-images is not given; the default --base-url "
             "hosts 10 (default: 10)"
    )

    parser.add_argument(
        "--large-memory",
        type=str,
        default="32 GB",
        help="Memory for preprocess_images and train_classifier, which hold "
             "the whole dataset in memory (default: 32 GB; '12 GB' fits the "
             "tutorial dataset at --image-size 128 on a 16 GB slot)"
    )

    # GPUs
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Request a GPU for the train_classifier job"
    )

    parser.add_argument(
        "--gpu-inference",
        action="store_true",
        help="Request a GPU for each per-image classify job (inference mode)"
    )

    parser.add_argument(
        "-s",
        "--hosted-site-catalog",
        metavar="FILE",
        type=str,
        default=None,
        help="Name of a Pegasus centrally hosted site catalog to plan against "
        "(e.g. access-pegasus.yml), instead of a locally generated one. Sets "
        "pegasus.catalog.site.repo.file; see "
        "https://pegasus.isi.edu/documentation/reference-guide/catalogs.html"
        "#centrally-hosted-site-catalogs",
    )
    parser.add_argument(
        "-e",
        "--execution-site-name",
        metavar="STR",
        type=str,
        default="compute",
        help="Execution site name (default: compute; condorpool on a plain "
        "HTCondor pool with no site catalog)",
    )

    # Container
    parser.add_argument(
        "--container-sif",
        type=str,
        default="Apptainer/CropHealth_Container.sif",
        help="Path to the Apptainer .sif image, absolute or relative to the "
             "workflow directory (default: Apptainer/CropHealth_Container.sif)"
    )

    # Output
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="workflow.yml",
        help="Output workflow file"
    )

    return parser


def generate(args, sites_catalog=False):
    """Validate args, then build and write the workflow and its catalogs.

    sites_catalog: also write the placeholder local + HTCondor site catalog
    (create_sites_catalog). The CLI never does; the notebook does when no
    hosted site catalog is used.
    """
    if args.mode != "inference" and args.data_source == "local" and not args.image_dir:
        raise ValueError("--image-dir required for local data source")
    if args.mode == "inference":
        if args.model_dir:
            missing = [
                f for f in ("disease_classifier.pt", "training_info.json")
                if not os.path.exists(os.path.join(args.model_dir, f))
            ]
            if missing:
                raise ValueError(
                    f"--model-dir {args.model_dir} is missing: {', '.join(missing)}"
                )
        if not args.inference_images and args.num_remote_images < 1:
            raise ValueError("--num-remote-images must be at least 1")

    workflow = CropHealthWorkflow(dagfile=args.output)

    workflow.create_pegasus_properties(hosted_site_catalog=args.hosted_site_catalog)
    if sites_catalog:
        workflow.create_sites_catalog(exec_site_name=args.execution_site_name)
    workflow.create_replica_catalog()
    workflow.create_transformation_catalog(
        exec_site_name=args.execution_site_name,
        container_sif=args.container_sif,
        gpu=args.gpu,
        large_memory=args.large_memory,
    )
    workflow.create_workflow(args)
    workflow.write()
    return workflow


def main():
    args = build_parser().parse_args()

    try:
        generate(args)

        logger.info("\n" + "=" * 70)
        logger.info("WORKFLOW GENERATION COMPLETE")
        logger.info("=" * 70)
        logger.info(f"  Workflow file: {args.output}")
        logger.info(f"  Mode: {args.mode}")
        if args.mode != "inference":
            logger.info(f"  Data source: {args.data_source}")
            logger.info(f"  Image size: {args.image_size}")
            logger.info(f"  Training epochs: {args.epochs}")
        logger.info("\nNext steps:")
        logger.info(f"  1. Review workflow: {args.output}")
        logger.info(f"  2. Plan and submit: pegasus-plan --dir submit -s {args.execution_site_name} -o local --submit {args.output}")
        logger.info(f"  3. Monitor status:  pegasus-status <submit_dir>")
        logger.info("=" * 70 + "\n")

    except Exception as e:
        logger.error(f"Failed to generate workflow: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
