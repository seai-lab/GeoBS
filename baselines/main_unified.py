#!/usr/bin/env python
"""
Unified main script for GeoBS evaluation.

This replaces the need for separate main_*.py files for each dataset/encoder combo.

Usage:
    python main_unified.py --dataset birdsnap --encoder space2vec_grid
    python main_unified.py --dataset nabirds --encoder space2vec_grid
    python main_unified.py --dataset birdsnap --encoder no_prior
    python main_unified.py --dataset inat_2018 --encoder xyz

The old way (still works but not recommended):
    python main_space2vec-grid_birdsnap_ebird.py
    python main_no_prior_birdsnap_ebird.py
    python main_space2vec-grid_nabirds_ebird.py
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd

from TorchSpatial.tester import test
from TorchSpatial.modules.encoder_selector import get_loc_encoder
from TorchSpatial.modules.models import LocationEncoder
import TorchSpatial.utils.datasets as data_import
from TorchSpatial.utils.config_loader import load_config, list_available_encoders, list_available_datasets
from TorchSpatial.utils.checkpoint_parser import update_config_from_checkpoint


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Unified GeoBS evaluation script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate Space2Vec-grid on Birdsnap
  python main_unified.py --dataset birdsnap --encoder space2vec_grid

  # Evaluate no_prior baseline on NABirds
  python main_unified.py --dataset nabirds --encoder no_prior

  # List available options
  python main_unified.py --list-encoders
  python main_unified.py --list-datasets
        """
    )

    parser.add_argument(
        "--dataset",
        type=str,
        choices=["birdsnap", "nabirds", "inat_2017", "inat_2018", "fmow", "yfcc"],
        help="Dataset to evaluate on"
    )

    parser.add_argument(
        "--encoder",
        type=str,
        help="Location encoder type (e.g., space2vec_grid, no_prior, xyz, nerf)"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (auto-detected if not specified)"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run on"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Batch size for evaluation"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="TorchSpatial/eval_results",
        help="Directory to save evaluation results"
    )

    parser.add_argument(
        "--list-encoders",
        action="store_true",
        help="List available encoder types and exit"
    )

    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="List available datasets and exit"
    )

    return parser.parse_args()


def auto_detect_checkpoint(dataset: str, encoder: str, checkpoint_dir: str = None, meta_type: str = None) -> str:
    """
    Auto-detect checkpoint path based on dataset and encoder.

    Args:
        dataset: Dataset name
        encoder: Encoder type
        checkpoint_dir: Directory containing checkpoints

    Returns:
        Path to checkpoint file

    Raises:
        FileNotFoundError: If no matching checkpoint found
    """
    # Auto-detect checkpoint directory
    if checkpoint_dir is None:
        # Try multiple possible locations
        possible_dirs = [
            Path("../TorchSpatial_checkpoint/data"),  # from baselines/
            Path("TorchSpatial_checkpoint/data"),     # from project root
            Path.cwd() / "TorchSpatial_checkpoint/data",
        ]

        checkpoint_path = None
        for d in possible_dirs:
            if d.exists():
                checkpoint_path = d
                break

        if checkpoint_path is None:
            raise FileNotFoundError(
                f"Could not find checkpoint directory. Tried:\n" +
                "\n".join([f"  - {d}" for d in possible_dirs]) +
                "\nPlease specify --checkpoint explicitly"
            )
    else:
        checkpoint_path = Path(checkpoint_dir)

    # Build search patterns
    # Normalize encoder name (case-insensitive)
    encoder_lower = encoder.lower()

    # Example patterns:
    # - With meta_type: model_birdsnap_ebird_meta_Space2Vec-grid_inception_v3_*.pth.tar
    # - Without meta_type: model_inat_2017_Space2Vec-grid_0.0100_*.pth.tar
    encoder_name_map = {
        "space2vec_grid": "Space2Vec-grid",
        "space2vec-grid": "Space2Vec-grid",
        "space2vec_theory": "Space2Vec-theory",
        "space2vec-theory": "Space2Vec-theory",
        "no_prior": "no_prior",
        "xyz": "xyz",
        "nerf": "NeRF",
        "sphere2vec": "Sphere2Vec",
        "sphere2vec_spherec": "Sphere2Vec-sphereC",
        "sphere2vec_spherem": "Sphere2Vec-sphereM",
        "sphere2vec_spherem+": "Sphere2Vec-sphereM+",
        "sphere2vec_dfs": "Sphere2Vec-dfs",
        "tile_ffn": "tile_ffn",
        "wrap": "wrap",
        "rbf": "rbf",
        "rff": "rff",
        "siren": "spherical_harmonics",
        "siren(sh)": "spherical_harmonics",
    }

    encoder_display = encoder_name_map.get(encoder_lower, encoder)

    # Try multiple patterns to handle different naming conventions
    # If meta_type is specified, prioritize matching it
    patterns = []

    if meta_type:
        # Prioritize patterns with specific meta_type
        patterns.extend([
            f"model_{dataset}_{meta_type}_{encoder_display}_*.pth.tar",     # Exact meta_type match
            f"model_{dataset}_{meta_type}_{encoder_display.lower()}_*.pth.tar",
        ])

    # Fallback patterns (match any or no meta_type)
    patterns.extend([
        f"model_{dataset}_*_{encoder_display}_*.pth.tar",      # With any meta_type
        f"model_{dataset}_{encoder_display}_*.pth.tar",        # Without meta_type
        f"model_{dataset}_*_{encoder_display.lower()}_*.pth.tar",
        f"model_{dataset}_{encoder_display.lower()}_*.pth.tar",
    ])

    matches = []
    for pattern in patterns:
        matches = list(checkpoint_path.glob(pattern))
        if matches:
            break

    if not matches:
        # List available checkpoints for this dataset
        all_checkpoints = list(checkpoint_path.glob(f"model_{dataset}_*.pth.tar"))
        available = "\n".join([f"  - {c.name}" for c in all_checkpoints[:5]])
        raise FileNotFoundError(
            f"No checkpoint found for: {dataset} + {encoder}\n"
            f"Tried patterns: {patterns}\n"
            f"Available checkpoints for {dataset}:\n{available}\n"
            f"Please specify --checkpoint explicitly"
        )

    if len(matches) > 1:
        print(f"⚠️  Multiple checkpoints found, using: {matches[0]}")

    return str(matches[0])


def main():
    """Main evaluation pipeline."""
    args = parse_args()

    # Handle list commands
    if args.list_encoders:
        encoders = list_available_encoders()
        print("Available encoders:")
        for enc in encoders:
            print(f"  - {enc}")
        return

    if args.list_datasets:
        datasets = list_available_datasets()
        print("Available datasets:")
        for ds in datasets:
            print(f"  - {ds}")
        return

    # Validate required arguments
    if not args.dataset or not args.encoder:
        print("❌ Error: --dataset and --encoder are required")
        print("   Use --help for usage information")
        return 1

    print("=" * 70)
    print(f"GeoBS Evaluation: {args.dataset} + {args.encoder}")
    print("=" * 70)

    # Load unified configuration
    print(f"\n📂 Loading configuration...")
    config = load_config(args.dataset, args.encoder)
    config["device"] = args.device

    # Get dataset-specific settings
    dataset_config = config[args.dataset]
    dataset_params = dataset_config["params"]
    num_classes = dataset_config["num_classes"]

    print(f"   Dataset: {args.dataset}")
    print(f"   Encoder: {config['loc_encoder_name']}")
    print(f"   Classes: {num_classes}")

    # Auto-detect or use specified checkpoint
    if config["loc_encoder_name"] != "no_prior":
        if args.checkpoint:
            checkpoint_path = args.checkpoint
        else:
            print(f"\n🔍 Auto-detecting checkpoint...")
            # Extract meta_type from dataset params if available
            meta_type = dataset_params.get("meta_type", None)
            checkpoint_path = auto_detect_checkpoint(args.dataset, args.encoder, meta_type=meta_type)

        print(f"   Checkpoint: {checkpoint_path}")

        # Update config with checkpoint-specific hyperparameters
        print(f"\n⚙️  Parsing checkpoint hyperparameters...")
        config = update_config_from_checkpoint(config, checkpoint_path)
        if 'loc_encoder_params' in config and config['loc_encoder_params']:
            params = config['loc_encoder_params']

            # Print appropriate parameter based on encoder type
            if 'siren' in config['loc_encoder_name'].lower() or 'spherical' in config['loc_encoder_name'].lower():
                print(f"   legendre_poly_num: {params.get('legendre_poly_num')}")
            else:
                print(f"   frequency_num: {params.get('frequency_num')}")

            print(f"   hidden_dim: {params.get('ffn_hidden_dim')}")
            print(f"   num_layers: {params.get('ffn_num_hidden_layers')}")

            if 'sphere2vec' in config['loc_encoder_name'].lower():
                print(f"   min_radius: {params.get('min_radius')}")
                print(f"   max_radius: {params.get('max_radius')}")
    else:
        checkpoint_path = None
        print(f"   Mode: No prior baseline (image-only)")

    # Load data
    print(f"\n📊 Loading dataset...")

    # For no_prior baseline, include all samples even without valid locations
    # For location encoders, only include samples with valid locations
    eval_remove_invalid = dataset_config["eval_remove_invalid"]
    if config["loc_encoder_name"] == "no_prior":
        eval_remove_invalid = False
        print(f"   Note: no_prior mode - including all samples (even without locations)")

    all_data = data_import.load_dataset(
        params=dataset_params,
        eval_split=config["eval_split"],
        train_remove_invalid=dataset_config["train_remove_invalid"],
        eval_remove_invalid=eval_remove_invalid,
        load_cnn_predictions=True,
        load_cnn_features=False,
        load_cnn_features_train=False
    )

    print(f"   Train samples: {len(all_data['train_classes'])}")
    print(f"   Test samples: {len(all_data['val_classes'])}")

    # Create DataLoader
    img_te = torch.Tensor(all_data["val_preds"])
    loc_te = torch.Tensor(all_data["val_locs"])
    y_te = torch.Tensor(all_data["val_classes"]).long()
    idx_te = np.arange(img_te.shape[0])

    test_data_zip = list(zip(idx_te, img_te, loc_te, y_te))
    test_loader = DataLoader(test_data_zip, batch_size=args.batch_size, shuffle=False)

    # Load location encoder (if not no_prior)
    loc_encoder = None
    if config["loc_encoder_name"] != "no_prior":
        print(f"\n🏗️  Building location encoder...")
        loc_encoder_params = config["loc_encoder_params"]
        loc_encoder_params["device"] = args.device

        # Get the appropriate encoder
        spa_enc = get_loc_encoder(
            name=config["loc_encoder_name"],
            overrides=loc_encoder_params
        ).to(args.device)

        # Get num_users from data or checkpoint
        # First, try to load checkpoint to get the correct num_users
        print(f"   Loading checkpoint to detect num_users...")
        checkpoint = torch.load(checkpoint_path, map_location=args.device)

        # Extract num_users from checkpoint's user_emb.weight shape
        if "state_dict" in checkpoint and "user_emb.weight" in checkpoint["state_dict"]:
            num_users = checkpoint["state_dict"]["user_emb.weight"].shape[0]
            print(f"   Detected num_users from checkpoint: {num_users}")
        else:
            num_users = all_data.get("num_users", 1)
            print(f"   Using num_users from data: {num_users}")

        # Wrap in LocationEncoder
        loc_encoder = LocationEncoder(
            spa_enc=spa_enc,
            num_inputs=2,
            num_classes=num_classes,
            num_filts=256,
            num_users=num_users,
        ).to(args.device)

        # Load checkpoint weights
        print(f"   Loading model weights...")
        loc_encoder.load_state_dict(checkpoint["state_dict"])

        loc_encoder.eval()
        print(f"   ✅ Model loaded (trained {checkpoint.get('epoch', 0) - 1} epochs)")

    # Run evaluation
    print(f"\n🚀 Running evaluation on test set...")
    rows = test(test_loader, loc_encoder)

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"eval_{args.dataset}_{args.encoder}.csv"
    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False)

    print(f"\n💾 Results saved to: {output_file}")

    # Print summary
    print(f"\n" + "=" * 70)
    print(f"Evaluation Complete!")
    print(f"=" * 70)

    return 0


if __name__ == "__main__":
    exit(main())
