#!/usr/bin/env python3

"""
Merge per-image prediction files into one predictions.json.

In inference mode the workflow runs one classify_disease job per image. This
step gathers their outputs into the single predictions file that
generate_report expects, recomputing the summary over all images.

Usage:
    ./merge_predictions.py --output predictions.json \
        00_predictions.json 01_predictions.json ...
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def summarize(results):
    """Build the summary and critical alerts (same shape as classify_disease)."""
    summary = {
        'total': len(results),
        'healthy': sum(1 for r in results if r.get('is_healthy', False)),
        'diseased': sum(1 for r in results if not r.get('is_healthy', True) and 'error' not in r),
        'errors': sum(1 for r in results if 'error' in r),
    }

    disease_counts = {}
    for r in results:
        if 'error' not in r:
            disease = r.get('disease', 'Unknown')
            disease_counts[disease] = disease_counts.get(disease, 0) + 1
    summary['disease_breakdown'] = disease_counts

    critical = [r for r in results if r.get('treatment', {}).get('severity') == 'critical']
    alerts = [
        {
            'image': r['filename'],
            'disease': r['disease'],
            'action': r['treatment']['action'],
        }
        for r in critical
    ]
    return summary, alerts


def merge(input_files, output_file):
    """Merge prediction files; return the number of successful predictions."""
    results = []
    framework = None
    for path in input_files:
        try:
            with open(path, 'r') as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Skipping unreadable predictions file {path}: {e}")
            continue
        results.extend(data.get('predictions', []))
        framework = framework or data.get('metadata', {}).get('framework')

    summary, alerts = summarize(results)
    output = {
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'framework': framework,
            'total_images': len(results),
            'merged_from': [Path(p).name for p in input_files],
        },
        'predictions': results,
        'summary': summary,
    }
    if alerts:
        output['critical_alerts'] = alerts

    successful = len(results) - summary['errors']
    if not successful:
        output['error'] = (
            f"none of the {len(input_files)} input file(s) contained a "
            "successful prediction"
        )

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    logger.info(f"Merged {len(results)} predictions from {len(input_files)} files "
                f"into {output_file}")
    logger.info(f"Summary: {summary['healthy']} healthy, {summary['diseased']} diseased, "
                f"{summary['errors']} errors")
    return successful


def main():
    parser = argparse.ArgumentParser(
        description="Merge per-image prediction files"
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Merged predictions JSON file'
    )

    parser.add_argument(
        'inputs',
        nargs='+',
        help='Per-image predictions JSON files'
    )

    args = parser.parse_args()

    # The merged file is written on every path (Pegasus declared it, and a
    # missing declared output holds the job instead of failing it), then the
    # step fails if there is nothing for the report to show.
    if not merge(args.inputs, args.output):
        logger.error(f"No successful predictions; wrote {args.output} with the reason "
                     "and failing this step.")
        sys.exit(1)


if __name__ == "__main__":
    main()
