#!/usr/bin/env python
"""
Complete evaluation suite for GeoBS unified system.

Tests all encoder × dataset combinations and generates result tables.

Usage:
    python run_full_evaluation.py
    python run_full_evaluation.py --encoders space2vec_grid nerf
    python run_full_evaluation.py --datasets birdsnap nabirds
"""

import argparse
import sys
import time
from pathlib import Path
from datetime import datetime
import pandas as pd

# Add baselines to path
sys.path.insert(0, str(Path(__file__).parent))

from TorchSpatial.utils.config_loader import list_available_encoders, list_available_datasets


# ============================================================================
# TABLE ORDER CONFIGURATION
# Customize these lists to control the order of rows and columns in results
# ============================================================================

# Row order (encoders) - from top to bottom
ENCODER_ORDER = [
    'no_prior',
    'tile_ffn',
    'space2vec_grid',
    'space2vec_theory',
    'nerf',
]

# Column order (datasets) - from left to right
DATASET_ORDER = [
    'birdsnap',
    'nabirds',
    'inat_2017',
    'inat_2018',
    'yfcc',
    'fmow',
]

# ============================================================================


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run complete GeoBS evaluation")

    parser.add_argument(
        '--encoders',
        nargs='+',
        default=None,
        help='Encoders to test (default: all in ENCODER_ORDER)'
    )

    parser.add_argument(
        '--datasets',
        nargs='+',
        default=None,
        help='Datasets to test (default: all in DATASET_ORDER)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='evaluation_results',
        help='Directory to save results'
    )

    parser.add_argument(
        '--skip-errors',
        action='store_true',
        help='Continue on errors instead of stopping'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda'],
        help='Device to run on'
    )

    return parser.parse_args()


def run_single_evaluation(dataset, encoder, device='cpu'):
    """
    Run evaluation for a single dataset × encoder combination.

    Returns:
        dict with keys: top1_acc, top3_acc, mrr, num_samples, time_elapsed
        or None if failed
    """
    print(f"\n{'='*70}")
    print(f"Evaluating: {dataset} × {encoder}")
    print(f"{'='*70}")

    start_time = time.time()

    try:
        # Import here to avoid circular imports
        from main_unified import main as run_evaluation_main

        # Override sys.argv to pass arguments
        old_argv = sys.argv
        sys.argv = [
            'main_unified.py',
            '--dataset', dataset,
            '--encoder', encoder,
            '--device', device,
        ]

        # Run evaluation
        run_evaluation_main()

        # Restore argv
        sys.argv = old_argv

        elapsed = time.time() - start_time

        # Parse results from the CSV file
        output_dir = Path('TorchSpatial/eval_results')
        result_file = output_dir / f'eval_{dataset}_{encoder}.csv'

        if result_file.exists():
            df = pd.read_csv(result_file)

            results = {
                'top1_acc': df['hit@1'].mean() * 100,
                'top3_acc': df['hit@3'].mean() * 100,
                'mrr': df['reciprocal_rank'].mean(),
                'num_samples': len(df),
                'time_elapsed': elapsed,
                'status': 'success'
            }

            print(f"\n✅ Success: Top-1={results['top1_acc']:.2f}%, "
                  f"Top-3={results['top3_acc']:.2f}%, "
                  f"MRR={results['mrr']:.4f}")

            return results
        else:
            print(f"\n⚠️  Warning: Result file not found")
            return {
                'top1_acc': None,
                'top3_acc': None,
                'mrr': None,
                'num_samples': None,
                'time_elapsed': elapsed,
                'status': 'not_found'
            }

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n❌ Failed: {str(e)[:100]}")

        return {
            'top1_acc': None,
            'top3_acc': None,
            'mrr': None,
            'num_samples': None,
            'time_elapsed': elapsed,
            'status': 'failed',
            'error': str(e)[:200]
        }
    finally:
        # Always restore argv
        sys.argv = old_argv


def create_results_tables(results_dict, encoders, datasets):
    """
    Create three result tables with specified order.

    Rows: encoders, Columns: datasets

    Args:
        results_dict: dict of {(dataset, encoder): results}
        encoders: list of encoders in desired order (rows)
        datasets: list of datasets in desired order (columns)

    Returns:
        Three DataFrames: top1_table, top3_table, mrr_table
    """
    # Initialize data structures
    top1_data = []
    top3_data = []
    mrr_data = []

    # Iterate through encoders (rows)
    for encoder in encoders:
        top1_row = {'Encoder': encoder}
        top3_row = {'Encoder': encoder}
        mrr_row = {'Encoder': encoder}

        # Iterate through datasets (columns)
        for dataset in datasets:
            result = results_dict.get((dataset, encoder))

            if result and result['status'] == 'success':
                top1_row[dataset] = f"{result['top1_acc']:.2f}"
                top3_row[dataset] = f"{result['top3_acc']:.2f}"
                mrr_row[dataset] = f"{result['mrr']:.4f}"
            elif result and result['status'] == 'failed':
                top1_row[dataset] = "FAIL"
                top3_row[dataset] = "FAIL"
                mrr_row[dataset] = "FAIL"
            else:
                top1_row[dataset] = "N/A"
                top3_row[dataset] = "N/A"
                mrr_row[dataset] = "N/A"

        top1_data.append(top1_row)
        top3_data.append(top3_row)
        mrr_data.append(mrr_row)

    # Create DataFrames with specified column order
    columns = ['Encoder'] + datasets

    top1_df = pd.DataFrame(top1_data, columns=columns)
    top3_df = pd.DataFrame(top3_data, columns=columns)
    mrr_df = pd.DataFrame(mrr_data, columns=columns)

    return top1_df, top3_df, mrr_df


def print_table(df, title):
    """Print a formatted table."""
    print(f"\n{'='*70}")
    print(f"{title}")
    print(f"{'='*70}")
    print(df.to_string(index=False))


def save_results(top1_df, top3_df, mrr_df, output_dir, timestamp=None):
    """Save results to files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save CSV files
    top1_df.to_csv(output_path / f'top1_accuracy_{timestamp}.csv', index=False)
    top3_df.to_csv(output_path / f'top3_accuracy_{timestamp}.csv', index=False)
    mrr_df.to_csv(output_path / f'mrr_{timestamp}.csv', index=False)

    # Save Markdown file with all three tables
    md_file = output_path / f'results_{timestamp}.md'
    with open(md_file, 'w') as f:
        f.write(f"# GeoBS Evaluation Results\n\n")
        f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write(f"## Top-1 Accuracy (%)\n\n")
        f.write(top1_df.to_markdown(index=False))
        f.write(f"\n\n")

        f.write(f"## Top-3 Accuracy (%)\n\n")
        f.write(top3_df.to_markdown(index=False))
        f.write(f"\n\n")

        f.write(f"## Mean Reciprocal Rank (MRR)\n\n")
        f.write(mrr_df.to_markdown(index=False))
        f.write(f"\n")

    print(f"\n📁 Results saved to: {output_path}/")
    print(f"   - top1_accuracy_{timestamp}.csv")
    print(f"   - top3_accuracy_{timestamp}.csv")
    print(f"   - mrr_{timestamp}.csv")
    print(f"   - results_{timestamp}.md")

    return output_path


def main():
    """Main evaluation loop."""
    args = parse_args()

    print("="*70)
    print("GeoBS Complete Evaluation Suite")
    print("="*70)

    # Determine which datasets and encoders to test
    # Use specified order from configuration
    if args.datasets:
        # Filter to only requested datasets, but keep the order
        datasets = [d for d in DATASET_ORDER if d in args.datasets]
    else:
        # Use all datasets in specified order
        datasets = DATASET_ORDER.copy()

    if args.encoders:
        # Filter to only requested encoders, but keep the order
        encoders = [e for e in ENCODER_ORDER if e in args.encoders]
    else:
        # Use all encoders in specified order
        encoders = ENCODER_ORDER.copy()

    print(f"\n📊 Evaluation Plan:")
    print(f"   Encoders (rows): {', '.join(encoders)}")
    print(f"   Datasets (cols): {', '.join(datasets)}")
    print(f"   Total combinations: {len(encoders) * len(datasets)}")
    print(f"   Device: {args.device}")

    # Confirm
    try:
        input(f"\n⏸️  Press Enter to start evaluation...")
    except EOFError:
        # Non-interactive mode
        print(f"\n🚀 Starting evaluation (non-interactive mode)...")

    # Run evaluations
    results = {}
    total = len(encoders) * len(datasets)
    completed = 0
    successful = 0
    failed = 0

    start_time = time.time()

    for dataset in datasets:
        for encoder in encoders:
            completed += 1
            print(f"\n{'='*70}")
            print(f"Progress: {completed}/{total} ({completed/total*100:.1f}%)")
            print(f"{'='*70}")

            result = run_single_evaluation(dataset, encoder, args.device)
            results[(dataset, encoder)] = result

            if result and result['status'] == 'success':
                successful += 1
            elif result and result['status'] == 'failed':
                failed += 1
                if not args.skip_errors:
                    print(f"\n❌ Stopping due to error. Use --skip-errors to continue on failures.")
                    break

        if not args.skip_errors and failed > 0:
            break

    total_time = time.time() - start_time

    # Create result tables with specified order
    print(f"\n{'='*70}")
    print("Creating result tables...")
    print(f"{'='*70}")

    top1_df, top3_df, mrr_df = create_results_tables(results, encoders, datasets)

    # Print tables
    print_table(top1_df, "TOP-1 ACCURACY (%)")
    print_table(top3_df, "TOP-3 ACCURACY (%)")
    print_table(mrr_df, "MEAN RECIPROCAL RANK (MRR)")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = save_results(top1_df, top3_df, mrr_df, args.output_dir, timestamp)

    # Print summary
    print(f"\n{'='*70}")
    print("EVALUATION COMPLETE")
    print(f"{'='*70}")
    print(f"   Total: {completed}/{total}")
    print(f"   Successful: {successful}")
    print(f"   Failed: {failed}")
    print(f"   Total time: {total_time/60:.1f} minutes")
    print(f"{'='*70}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
