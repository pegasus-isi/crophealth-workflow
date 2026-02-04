#!/usr/bin/env python3

"""
Fetch and organize crop health images for disease detection workflow.

Supports multiple data sources:
1. Local directory of images
2. Kaggle PlantVillage dataset (requires kaggle API key)
3. Sample images from URL for testing

Usage:
    # Use local images
    ./fetch_crop_images.py --source local --input-dir ./images --output crops.csv

    # Download from Kaggle
    ./fetch_crop_images.py --source kaggle --dataset emmarex/plantdisease \
        --output crops.csv

    # Fetch sample test images
    ./fetch_crop_images.py --source sample --output crops.csv
"""

import argparse
import csv
import json
import logging
import os
import shutil
import sys
import tarfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse
import kagglehub

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Supported crop diseases and their descriptions
DISEASE_INFO = {
    # Apple
    'Apple___Apple_scab': {'crop': 'Apple', 'disease': 'Apple Scab', 'treatment': 'Fungicide application, remove infected leaves'},
    'Apple___Black_rot': {'crop': 'Apple', 'disease': 'Black Rot', 'treatment': 'Remove infected fruit, apply fungicide'},
    'Apple___Cedar_apple_rust': {'crop': 'Apple', 'disease': 'Cedar Apple Rust', 'treatment': 'Remove nearby cedar trees, fungicide'},
    'Apple___healthy': {'crop': 'Apple', 'disease': 'Healthy', 'treatment': 'No treatment needed'},

    # Corn
    'Corn_(maize)___Cercospora_leaf_spot': {'crop': 'Corn', 'disease': 'Gray Leaf Spot', 'treatment': 'Resistant varieties, fungicide'},
    'Corn_(maize)___Common_rust': {'crop': 'Corn', 'disease': 'Common Rust', 'treatment': 'Fungicide, plant resistant hybrids'},
    'Corn_(maize)___Northern_Leaf_Blight': {'crop': 'Corn', 'disease': 'Northern Leaf Blight', 'treatment': 'Crop rotation, fungicide'},
    'Corn_(maize)___healthy': {'crop': 'Corn', 'disease': 'Healthy', 'treatment': 'No treatment needed'},

    # Grape
    'Grape___Black_rot': {'crop': 'Grape', 'disease': 'Black Rot', 'treatment': 'Remove infected tissue, fungicide'},
    'Grape___Esca_(Black_Measles)': {'crop': 'Grape', 'disease': 'Esca', 'treatment': 'Prune infected wood, no cure'},
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': {'crop': 'Grape', 'disease': 'Leaf Blight', 'treatment': 'Fungicide application'},
    'Grape___healthy': {'crop': 'Grape', 'disease': 'Healthy', 'treatment': 'No treatment needed'},

    # Potato
    'Potato___Early_blight': {'crop': 'Potato', 'disease': 'Early Blight', 'treatment': 'Fungicide, crop rotation'},
    'Potato___Late_blight': {'crop': 'Potato', 'disease': 'Late Blight', 'treatment': 'Destroy infected plants, fungicide'},
    'Potato___healthy': {'crop': 'Potato', 'disease': 'Healthy', 'treatment': 'No treatment needed'},

    # Tomato
    'Tomato___Bacterial_spot': {'crop': 'Tomato', 'disease': 'Bacterial Spot', 'treatment': 'Copper-based bactericide'},
    'Tomato___Early_blight': {'crop': 'Tomato', 'disease': 'Early Blight', 'treatment': 'Fungicide, remove lower leaves'},
    'Tomato___Late_blight': {'crop': 'Tomato', 'disease': 'Late Blight', 'treatment': 'Destroy infected plants, fungicide'},
    'Tomato___Leaf_Mold': {'crop': 'Tomato', 'disease': 'Leaf Mold', 'treatment': 'Improve ventilation, fungicide'},
    'Tomato___Septoria_leaf_spot': {'crop': 'Tomato', 'disease': 'Septoria Leaf Spot', 'treatment': 'Remove infected leaves, fungicide'},
    'Tomato___Spider_mites': {'crop': 'Tomato', 'disease': 'Spider Mites', 'treatment': 'Miticide, increase humidity'},
    'Tomato___Target_Spot': {'crop': 'Tomato', 'disease': 'Target Spot', 'treatment': 'Fungicide application'},
    'Tomato___Yellow_Leaf_Curl_Virus': {'crop': 'Tomato', 'disease': 'Yellow Leaf Curl Virus', 'treatment': 'Remove infected plants, control whiteflies'},
    'Tomato___mosaic_virus': {'crop': 'Tomato', 'disease': 'Mosaic Virus', 'treatment': 'Remove infected plants, sanitize tools'},
    'Tomato___healthy': {'crop': 'Tomato', 'disease': 'Healthy', 'treatment': 'No treatment needed'},
}


def scan_local_images(input_dir: str, output_dir: str, recursive: bool = False) -> List[Dict]:
    """
    Scan local directory for images and create catalog.

    Expected structure:
        input_dir/
            Crop___Disease/
                image1.jpg
                image2.jpg
            Crop___healthy/
                image3.jpg
    """
    records = []
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        return records

    # Scan for disease categories
    category_dirs = []
    if recursive:
        category_dirs = sorted(
            p for p in input_path.rglob("*")
            if p.is_dir() and "___" in p.name
        )
    else:
        category_dirs = sorted(p for p in input_path.iterdir() if p.is_dir())

    for category_dir in category_dirs:
        if not category_dir.is_dir():
            continue

        category_name = category_dir.name
        disease_info = DISEASE_INFO.get(category_name, {
            'crop': category_name.split('___')[0] if '___' in category_name else 'Unknown',
            'disease': category_name.split('___')[1] if '___' in category_name else 'Unknown',
            'treatment': 'Consult agricultural expert',
        })

        # Scan images in category
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
        for img_file in sorted(category_dir.iterdir()):
            if img_file.suffix.lower() not in image_extensions:
                continue

            # Copy to output directory with category prefix
            output_filename = f"{category_name}_{img_file.name}"
            output_filepath = output_path / output_filename
            shutil.copy2(img_file, output_filepath)

            records.append({
                'image_path': str(output_filepath),
                'filename': output_filename,
                'category': category_name,
                'crop': disease_info['crop'],
                'disease': disease_info['disease'],
                'treatment': disease_info['treatment'],
                'is_healthy': disease_info['disease'] == 'Healthy',
            })

    logger.info(f"Found {len(records)} images in {len(set(r['category'] for r in records))} categories")
    return records


def download_kaggle_dataset(dataset: str, output_dir: str) -> List[Dict]:
    """
    Download dataset from Kaggle.

    Requires:
        - kaggle package installed
        - kaggle credentials via kaggle.json or environment variables
    """
    def parse_kaggle_api_token(token: str) -> Optional[Tuple[str, str]]:
        try:
            data = json.loads(token)
        except json.JSONDecodeError:
            data = None

        if isinstance(data, dict) and data.get('username') and data.get('key'):
            return data['username'], data['key']

        for separator in (':', ',', '|'):
            if separator in token:
                username, key = token.split(separator, 1)
                return username.strip(), key.strip()

        return None

    def ensure_kaggle_config() -> bool:
        config_dir = Path(os.environ.get('KAGGLE_CONFIG_DIR', '~/.config/kaggle')).expanduser()
        config_file = config_dir / 'kaggle.json'

        if config_file.exists():
            return True

        username = os.environ.get('KAGGLE_USERNAME')
        key = os.environ.get('KAGGLE_KEY')
        if username and key:
            config_dir.mkdir(parents=True, exist_ok=True)
            with open(config_file, 'w') as f:
                json.dump({'username': username, 'key': key}, f)
            try:
                os.chmod(config_file, 0o600)
            except PermissionError:
                logger.warning("Unable to set permissions on kaggle.json")
            return True

        token = os.environ.get('KAGGLE_API_TOKEN')
        if token:
            parsed = parse_kaggle_api_token(token)
            if parsed:
                username, key = parsed
                config_dir.mkdir(parents=True, exist_ok=True)
                with open(config_file, 'w') as f:
                    json.dump({'username': username, 'key': key}, f)
                try:
                    os.chmod(config_file, 0o600)
                except PermissionError:
                    logger.warning("Unable to set permissions on kaggle.json")
                return True

            logger.error("KAGGLE_API_TOKEN format not recognized. Expected JSON or 'username:key'.")
            return False

        logger.error("Kaggle credentials not found.")
        logger.error("Set KAGGLE_USERNAME/KAGGLE_KEY, or KAGGLE_API_TOKEN with JSON or 'username:key'.")
        logger.error("Get your credentials from: https://www.kaggle.com/settings → API → Create New Token")
        return False

    if not ensure_kaggle_config():
        return []

    logger.info(f"Downloading dataset via kagglehub: {dataset}")
    try:
        dataset_path = kagglehub.dataset_download(dataset)
    except Exception as e:
        logger.error(f"KaggleHub download failed: {e}")
        return []
    return scan_local_images(str(dataset_path), output_dir, recursive=True)


def create_sample_catalog(output_dir: str) -> List[Dict]:
    """
    Create sample catalog with placeholder entries for testing.

    In production, this would download sample images from a public URL.
    """
    records = []
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create sample entries for different disease types
    sample_categories = [
        'Tomato___Early_blight',
        'Tomato___Late_blight',
        'Tomato___healthy',
        'Potato___Early_blight',
        'Potato___healthy',
        'Corn_(maize)___Common_rust',
        'Corn_(maize)___healthy',
    ]

    for i, category in enumerate(sample_categories):
        disease_info = DISEASE_INFO.get(category, {
            'crop': 'Unknown',
            'disease': 'Unknown',
            'treatment': 'Consult expert',
        })

        # Create placeholder entries (would be actual images in production)
        for j in range(3):
            filename = f"{category}_sample_{j+1}.jpg"
            records.append({
                'image_path': str(output_path / filename),
                'filename': filename,
                'category': category,
                'crop': disease_info['crop'],
                'disease': disease_info['disease'],
                'treatment': disease_info['treatment'],
                'is_healthy': disease_info['disease'] == 'Healthy',
                'is_sample': True,  # Mark as sample data
            })

    logger.info(f"Created {len(records)} sample catalog entries")
    logger.warning("Sample mode: No actual images downloaded. Use --source local or kaggle for real data.")
    return records


def save_catalog(records: List[Dict], output_file: str):
    """Save image catalog to CSV."""
    if not records:
        logger.warning("No records to save")
        return

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ['image_path', 'filename', 'category', 'crop', 'disease', 'treatment', 'is_healthy']
    if records[0].get('is_sample'):
        fieldnames.append('is_sample')

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            # Only write fields that exist in fieldnames
            row = {k: v for k, v in record.items() if k in fieldnames}
            writer.writerow(row)

    logger.info(f"Saved catalog to {output_file}")

    # Print summary
    crops = set(r['crop'] for r in records)
    diseases = set(r['disease'] for r in records)
    healthy = sum(1 for r in records if r['is_healthy'])
    diseased = len(records) - healthy

    logger.info(f"Summary:")
    logger.info(f"  Total images: {len(records)}")
    logger.info(f"  Crops: {', '.join(sorted(crops))}")
    logger.info(f"  Diseases: {len(diseases)} types")
    logger.info(f"  Healthy: {healthy}, Diseased: {diseased}")


def main():
    parser = argparse.ArgumentParser(
        description="Fetch and organize crop health images"
    )

    parser.add_argument(
        '--source', '-s',
        type=str,
        choices=['local', 'kaggle', 'sample'],
        default='sample',
        help='Data source (local, kaggle, or sample for testing)'
    )

    parser.add_argument(
        '--input-dir', '-i',
        type=str,
        help='Input directory for local images'
    )

    parser.add_argument(
        '--dataset',
        type=str,
        default='emmarex/plantdisease',
        help='Kaggle dataset name (for kaggle source)'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        default='crop_catalog.csv',
        help='Output catalog CSV file'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='./images',
        help='Output directory for processed images'
    )

    parser.add_argument(
        '--archive-output',
        type=str,
        default=None,
        help='Optional tar.gz archive of output images'
    )

    args = parser.parse_args()

    if args.source == 'local':
        if not args.input_dir:
            logger.error("--input-dir required for local source")
            sys.exit(1)
        records = scan_local_images(args.input_dir, args.output_dir)

    elif args.source == 'kaggle':
        records = download_kaggle_dataset(args.dataset, args.output_dir)

    else:  # sample
        records = create_sample_catalog(args.output_dir)

    save_catalog(records, args.output)

    if args.archive_output:
        output_path = Path(args.output_dir)
        archive_path = Path(args.archive_output)
        with tarfile.open(archive_path, "w:gz") as tar:
            tar.add(output_path, arcname=output_path.name)
        logger.info(f"Archived images to {archive_path}")


if __name__ == "__main__":
    main()
