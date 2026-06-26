"""
Unified configuration loader to avoid duplicated config files.

Usage:
    from TorchSpatial.utils.config_loader import load_config

    # Load configuration for specific dataset and encoder
    config = load_config(dataset="birdsnap", encoder="space2vec_grid")

    # Or load with custom paths
    config = load_config(
        dataset="birdsnap",
        encoder="space2vec_grid",
        base_config_path="configs/base_config.json",
        encoder_config_dir="configs/encoders/"
    )

This replaces the need for separate configs_*.json files for each combination.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional


def load_config(
    dataset: str,
    encoder: str = "space2vec_grid",
    base_config_path: Optional[str] = None,
    encoder_config_dir: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Load unified configuration by merging base config with encoder-specific config.

    Args:
        dataset: Dataset name (e.g., "birdsnap", "nabirds", "inat_2018")
        encoder: Encoder type (e.g., "space2vec_grid", "no_prior", "xyz", "nerf")
        base_config_path: Path to base configuration file (auto-detected if None)
        encoder_config_dir: Directory containing encoder-specific configs (auto-detected if None)
        overrides: Optional dict of values to override in final config

    Returns:
        Merged configuration dictionary with structure compatible with old configs

    Example:
        >>> config = load_config("birdsnap", "space2vec_grid")
        >>> config["dataset"]  # "birdsnap"
        >>> config["loc_encoder_name"]  # "Space2Vec-grid"
        >>> config["birdsnap"]["num_classes"]  # 500
    """

    # Auto-detect config paths if not provided
    if base_config_path is None:
        possible_base_paths = [
            Path("configs/base_config.json"),
            Path("../configs/base_config.json"),
            Path(__file__).parent.parent.parent / "configs/base_config.json",
        ]
        base_path = None
        for p in possible_base_paths:
            if p.exists():
                base_path = p
                break
        if base_path is None:
            raise FileNotFoundError(
                f"Could not find base_config.json. Tried:\n" +
                "\n".join([f"  - {p}" for p in possible_base_paths])
            )
    else:
        base_path = Path(base_config_path)

    if encoder_config_dir is None:
        possible_encoder_dirs = [
            Path("configs/encoders/"),
            Path("../configs/encoders/"),
            Path(__file__).parent.parent.parent / "configs/encoders/",
        ]
        encoder_dir = None
        for d in possible_encoder_dirs:
            if d.exists():
                encoder_dir = d
                break
        if encoder_dir is None:
            raise FileNotFoundError(
                f"Could not find configs/encoders/. Tried:\n" +
                "\n".join([f"  - {d}" for d in possible_encoder_dirs])
            )
    else:
        encoder_dir = Path(encoder_config_dir)

    # Load base config (datasets and shared settings)
    if not base_path.exists():
        raise FileNotFoundError(f"Base config not found: {base_path}")

    with open(base_path, 'r') as f:
        config = json.load(f)

    # Load encoder-specific config (normalize to lowercase)
    encoder_normalized = encoder.lower()
    encoder_path = encoder_dir / f"{encoder_normalized}.json"
    if not encoder_path.exists():
        available = list_available_encoders(str(encoder_dir))
        raise FileNotFoundError(
            f"Encoder config not found: {encoder_path}\n"
            f"Available encoders: {available}"
        )

    with open(encoder_path, 'r') as f:
        encoder_config = json.load(f)

    # Merge encoder config into base (encoder-specific settings take precedence)
    config.update(encoder_config)

    # Set the active dataset
    config["dataset"] = dataset

    # Apply eval_split overrides for specific datasets
    if "eval_split_overrides" in config and dataset in config["eval_split_overrides"]:
        config["eval_split"] = config["eval_split_overrides"][dataset]

    # For backward compatibility: flatten dataset configs to top level
    # Old code expects config["birdsnap"], config["nabirds"], etc.
    if "datasets" in config:
        for ds_name, ds_config in config["datasets"].items():
            config[ds_name] = ds_config
        # Don't remove "datasets" key - some code might still use it

    # Apply overrides if provided
    if overrides:
        config.update(overrides)

    return config


def list_available_encoders(encoder_config_dir: str = "configs/encoders/") -> list:
    """
    List all available encoder configurations.

    Returns:
        List of encoder names (without .json extension)
    """
    encoder_dir = Path(encoder_config_dir)
    if not encoder_dir.exists():
        return []

    return [f.stem for f in encoder_dir.glob("*.json")]


def list_available_datasets(base_config_path: str = "configs/base_config.json") -> list:
    """
    List all available datasets from base config.

    Returns:
        List of dataset names
    """
    try:
        with open(base_config_path, 'r') as f:
            config = json.load(f)
        return list(config.get("datasets", {}).keys())
    except FileNotFoundError:
        return []


def create_legacy_config_file(
    dataset: str,
    encoder: str,
    output_path: str,
    **kwargs
) -> None:
    """
    Create a legacy-style config file for backward compatibility.

    This is useful if you need to generate old-style config files
    from the new unified system.

    Args:
        dataset: Dataset name
        encoder: Encoder type
        output_path: Where to save the generated config
        **kwargs: Additional overrides to include
    """
    config = load_config(dataset, encoder, overrides=kwargs)

    with open(output_path, 'w') as f:
        json.dump(config, f, indent=4)

    print(f"✅ Created legacy config: {output_path}")


# Example usage and testing
if __name__ == "__main__":
    print("=== Testing Unified Config Loader ===\n")

    # List available options
    print("Available encoders:", list_available_encoders())
    print("Available datasets:", list_available_datasets())
    print()

    # Load example configs
    examples = [
        ("birdsnap", "space2vec_grid"),
        ("nabirds", "space2vec_grid"),
        ("birdsnap", "no_prior"),
    ]

    for dataset, encoder in examples:
        try:
            config = load_config(dataset, encoder)
            print(f"✅ {dataset} + {encoder}:")
            print(f"   loc_encoder_name: {config.get('loc_encoder_name')}")
            print(f"   num_classes: {config[dataset]['num_classes']}")
            print()
        except Exception as e:
            print(f"❌ {dataset} + {encoder}: {e}\n")
