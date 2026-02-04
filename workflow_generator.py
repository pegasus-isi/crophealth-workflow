#!/usr/bin/env python3

"""
Crop Health Workflow Generator for Pegasus WMS.

Creates a workflow for:
1. Fetching/organizing crop images
2. Preprocessing images for ML
3. Training disease classification model
4. Running inference on new images
5. Generating health reports

Supports standard single-site execution and edge-to-cloud DPU architecture.

Usage:
    # Standard workflow with local images
    ./workflow_generator.py --image-dir ./field_images --output workflow.yml

    # Edge-to-cloud with DPU
    ./workflow_generator.py --image-dir ./field_images --enable-dpu \
        --edge-site edgepool --cloud-site cloudpool --output workflow.yml
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

from Pegasus.api import *

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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

    def create_pegasus_properties(self):
        """Create Pegasus properties configuration."""
        self.props = Properties()
        self.props["pegasus.transfer.threads"] = "16"

    def create_sites_catalog(self, exec_site_name="condorpool"):
        """Create site catalog."""
        logger.info(f"Creating site catalog for execution site: {exec_site_name}")
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

    def create_transformation_catalog(self, exec_site_name="condorpool", container_image="kthare10/crophealth:latest"):
        """Create transformation catalog with executables and containers."""
        logger.info("Creating transformation catalog")
        self.tc = TransformationCatalog()

        # Container - use Singularity with docker:// URL
        crophealth_container = Container(
            "crophealth_container",
            container_type=Container.SINGULARITY,
            image=f"docker://{container_image}",
            image_site="docker_hub",
        )

        # Add transformations
        fetch_crop_images = Transformation(
            "fetch_crop_images",
            site=exec_site_name,
            pfn=os.path.join(self.wf_dir, "fetch_crop_images.py"),
            is_stageable=True,
            container=crophealth_container,
        ).add_pegasus_profile(memory="4 GB")

        preprocess_images = Transformation(
            "preprocess_images",
            site=exec_site_name,
            pfn=os.path.join(self.wf_dir, "bin/preprocess_images.py"),
            is_stageable=True,
            container=crophealth_container,
        ).add_pegasus_profile(memory="32 GB")

        train_classifier = Transformation(
            "train_classifier",
            site=exec_site_name,
            pfn=os.path.join(self.wf_dir, "bin/train_classifier.py"),
            is_stageable=True,
            container=crophealth_container,
        ).add_pegasus_profile(memory="32 GB")

        classify_disease = Transformation(
            "classify_disease",
            site=exec_site_name,
            pfn=os.path.join(self.wf_dir, "bin/classify_disease.py"),
            is_stageable=True,
            container=crophealth_container,
        ).add_pegasus_profile(memory="32 GB")

        generate_report = Transformation(
            "generate_report",
            site=exec_site_name,
            pfn=os.path.join(self.wf_dir, "bin/generate_report.py"),
            is_stageable=True,
            container=crophealth_container,
        ).add_pegasus_profile(memory="32 GB")

        self.tc.add_containers(crophealth_container)
        self.tc.add_transformations(
            fetch_crop_images,
            preprocess_images,
            train_classifier,
            classify_disease,
            generate_report,
        )

    def create_workflow(self, args):
        """Create the complete workflow."""
        logger.info("Creating workflow")
        self.wf = Workflow(self.wf_name)

        # Output file names
        catalog_file = File("crop_catalog.csv")
        images_archive = File("images.tar.gz")
        train_data = File("train_data.npz")
        val_data = File("val_data.npz")
        label_mapping = File("label_mapping.json")
        preprocessing_info = File("preprocessing_info.json")
        model_checkpoint = File("disease_classifier.pt")
        training_info = File("training_info.json")
        predictions_file = File("predictions.json")
        report_html = File("report.html")
        report_summary = File("report_summary.json")
        disease_distribution_png = File("disease_distribution.png")
        severity_distribution_png = File("severity_distribution.png")
        crop_health_summary_png = File("crop_health_summary.png")
        confidence_histogram_png = File("confidence_histogram.png")

        # Job 1: Fetch/catalog images
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
        fetch_job.add_env(KAGGLE_USERNAME=os.environ.get('KAGGLE_USERNAME', ''))
        fetch_job.add_env(KAGGLE_KEY=os.environ.get('KAGGLE_KEY', ''))
        fetch_job.add_outputs(catalog_file, stage_out=True, register_replica=False)
        fetch_job.add_outputs(images_archive, stage_out=False, register_replica=False)
        fetch_job.add_pegasus_profile(label="fetch")

        # Job 2: Preprocess images
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

        # Job 3: Train classifier
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

        # Job 4: Classify diseases
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

        # Job 5: Generate report
        report_job = Job("generate_report", _id="report", node_label="report")
        report_job.add_args(
            "--predictions", predictions_file,
            "--output-dir", ".",
            "--format", "all",
        )
        report_job.add_inputs(predictions_file)
        report_job.add_outputs(report_html, stage_out=True, register_replica=False)
        report_job.add_outputs(report_summary, stage_out=True, register_replica=False)
        report_job.add_outputs(disease_distribution_png, stage_out=True, register_replica=False)
        report_job.add_outputs(severity_distribution_png, stage_out=True, register_replica=False)
        report_job.add_outputs(crop_health_summary_png, stage_out=True, register_replica=False)
        report_job.add_outputs(confidence_histogram_png, stage_out=True, register_replica=False)
        report_job.add_pegasus_profile(label="report")

        # Add jobs to workflow
        self.wf.add_jobs(fetch_job, preprocess_job, train_job, classify_job, report_job)

        # Define dependencies
        self.wf.add_dependency(fetch_job, children=[preprocess_job])
        self.wf.add_dependency(preprocess_job, children=[train_job])
        self.wf.add_dependency(train_job, children=[classify_job])
        self.wf.add_dependency(classify_job, children=[report_job])


def main():
    parser = argparse.ArgumentParser(
        description="Generate Pegasus workflow for crop disease detection"
    )

    # Data source
    parser.add_argument(
        "--data-source",
        type=str,
        choices=["local", "kaggle", "sample"],
        default="sample",
        help="Source of image data"
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

    # Execution site
    parser.add_argument(
        "-e", "--execution-site-name",
        type=str,
        default="condorpool",
        help="HTCondor pool name for execution"
    )

    # Container
    parser.add_argument(
        "--container-image",
        type=str,
        default="kthare10/crophealth:latest",
        help="Docker container image for workflow"
    )

    # Output
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="workflow.yml",
        help="Output workflow file"
    )

    args = parser.parse_args()

    # Validate
    if args.data_source == "local" and not args.image_dir:
        logger.error("--image-dir required for local data source")
        sys.exit(1)

    # Generate workflow
    try:
        workflow = CropHealthWorkflow(dagfile=args.output)
        workflow.create_pegasus_properties()
        workflow.create_sites_catalog(exec_site_name=args.execution_site_name)
        workflow.create_replica_catalog()
        workflow.create_transformation_catalog(
            exec_site_name=args.execution_site_name,
            container_image=args.container_image
        )
        workflow.create_workflow(args)
        workflow.write()

        logger.info("\n" + "=" * 70)
        logger.info("WORKFLOW GENERATION COMPLETE")
        logger.info("=" * 70)
        logger.info(f"  Workflow file: {args.output}")
        logger.info(f"  Data source: {args.data_source}")
        logger.info(f"  Image size: {args.image_size}")
        logger.info(f"  Training epochs: {args.epochs}")
        logger.info("\nNext steps:")
        logger.info(f"  1. Review workflow: {args.output}")
        logger.info(f"  2. Submit workflow: pegasus-plan --submit -s {args.execution_site_name} -o local {args.output}")
        logger.info(f"  3. Monitor status:  pegasus-status <submit_dir>")
        logger.info("=" * 70 + "\n")

    except Exception as e:
        logger.error(f"Failed to generate workflow: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
