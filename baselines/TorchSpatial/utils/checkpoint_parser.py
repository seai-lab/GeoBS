"""
Parse hyperparameters from checkpoint filenames.

Original TorchSpatial encodes hyperparameters in checkpoint filenames, e.g.:
model_birdsnap_ebird_meta_Space2Vec-grid_inception_v3_0.0100_128_0.1000000_360.000_1_512_leakyrelu.pth.tar

Format: model_{dataset}_{meta_type}_{encoder}_{cnn}_{lr}_{freq_num}_{min_radius}_{max_radius}_{num_layers}_{hidden_dim}_{activation}.pth.tar
"""

from pathlib import Path
from typing import Dict, Any, Optional


def parse_checkpoint_filename(checkpoint_path: str) -> Dict[str, Any]:
    """
    Extract hyperparameters from checkpoint filename.

    Args:
        checkpoint_path: Path to checkpoint file

    Returns:
        Dictionary of hyperparameters extracted from filename

    Example:
        >>> parse_checkpoint_filename("model_birdsnap_ebird_meta_Space2Vec-grid_inception_v3_0.0100_128_0.1000000_360.000_1_512_leakyrelu.pth.tar")
        {
            'dataset': 'birdsnap',
            'meta_type': 'ebird_meta',
            'encoder': 'Space2Vec-grid',
            'cnn': 'inception_v3',
            'lr': 0.01,
            'frequency_num': 128,
            'min_radius': 0.1,
            'max_radius': 360.0,
            'num_layers': 1,
            'hidden_dim': 512,
            'activation': 'leakyrelu'
        }
    """
    filename = Path(checkpoint_path).name  # Get filename
    # Remove .pth.tar extension (Path.stem only removes last extension)
    if filename.endswith('.pth.tar'):
        filename = filename[:-8]  # Remove '.pth.tar'
    elif filename.endswith('.tar'):
        filename = filename[:-4]  # Remove '.tar'

    # Try to parse standard format
    # model_{dataset}_{meta_type}_{encoder}_{cnn}_{lr}_{freq_num}_{min_radius}_{max_radius}_{num_layers}_{hidden_dim}_{activation}

    parts = filename.split('_')

    # Find encoder name (may contain hyphens like "Space2Vec-grid")
    encoder_idx = None
    encoder_name = None
    known_encoders = [
        'Space2Vec-grid', 'Space2Vec-theory',
        'Sphere2Vec-sphereC', 'Sphere2Vec-sphereM', 'Sphere2Vec-sphereM+', 'Sphere2Vec-dfs',
        'NeRF', 'xyz', 'Sphere2Vec',
        'spherical_harmonics',
        'wrap', 'rbf', 'rff', 'tile'
    ]

    # Special handling for multi-part encoder names (e.g., "spherical_harmonics")
    for i, part in enumerate(parts):
        # Check for "spherical_harmonics" (two parts)
        if i < len(parts) - 1 and part == 'spherical' and parts[i+1] == 'harmonics':
            encoder_name = 'spherical_harmonics'
            encoder_idx = i
            break

        # Standard single-part matching
        for enc in known_encoders:
            if enc.lower() in part.lower():
                # Reconstruct encoder name (may span multiple parts due to hyphen)
                if '-' in enc:
                    encoder_name = enc
                    encoder_idx = i
                else:
                    encoder_name = part
                    encoder_idx = i
                break
        if encoder_name:
            break

    if encoder_idx is None:
        # Fallback: assume structure without meta_type
        return {
            'dataset': parts[1] if len(parts) > 1 else None,
            'encoder': parts[2] if len(parts) > 2 else None,
        }

    result = {
        'dataset': parts[1] if len(parts) > 1 else None,
        'encoder': encoder_name,
    }

    # Meta type is between dataset and encoder (if exists)
    if encoder_idx > 2:
        result['meta_type'] = '_'.join(parts[2:encoder_idx])

    # CNN model follows encoder
    # For multi-part encoders like "spherical_harmonics", skip the second part
    if encoder_name == 'spherical_harmonics':
        cnn_idx = encoder_idx + 2  # Skip both "spherical" and "harmonics"
    else:
        cnn_idx = encoder_idx + 1

    if cnn_idx < len(parts) and 'inception' in parts[cnn_idx].lower():
        result['cnn'] = '_'.join(parts[cnn_idx:cnn_idx+2]) if cnn_idx+1 < len(parts) else parts[cnn_idx]
        cnn_idx += 1

    # After CNN model, numeric parameters follow
    numeric_start = cnn_idx + 1
    numeric_parts = []

    for i in range(numeric_start, len(parts)):
        part = parts[i]
        original_part = part  # For debugging

        # Skip BATCH* suffixes
        if 'BATCH' in part.upper():
            continue

        # Stop at activation function (check before removing suffix)
        part_lower = part.lower().replace('.pth.tar', '')
        if part_lower in ['leakyrelu', 'relu', 'sigmoid', 'tanh']:
            result['activation'] = part_lower
            break

        # Remove .pth.tar suffix if present
        if '.pth.tar' in part:
            part = part.replace('.pth.tar', '')

        # Try to parse as float
        try:
            val = float(part)
            numeric_parts.append(val)
            # print(f"DEBUG: Parsed {original_part} -> {val}")
        except ValueError:
            # print(f"DEBUG: Failed to parse {original_part} (cleaned: {part})")
            pass

    # Map numeric parts to parameters based on encoder type and expected order
    # Different encoders have different parameter orders

    # DEBUG: Uncomment to see parsed values
    # print(f"DEBUG: encoder_name={encoder_name}, numeric_parts={numeric_parts}, len={len(numeric_parts)}")

    if encoder_name and 'spherical' in encoder_name.lower():
        # Spherical Harmonics (SIREN) format: lr, frequency_num, min_param, num_layers, hidden_dim
        # Example: 0.0050_16_0.0001000_3_512 (no max_radius)
        if len(numeric_parts) >= 5:
            result['lr'] = numeric_parts[0]
            result['frequency_num'] = int(numeric_parts[1])
            # numeric_parts[2] is a threshold parameter, skip
            result['num_layers'] = int(numeric_parts[3])
            result['hidden_dim'] = int(numeric_parts[4])
        elif len(numeric_parts) >= 4:
            result['frequency_num'] = int(numeric_parts[0])
            result['num_layers'] = int(numeric_parts[2])
            result['hidden_dim'] = int(numeric_parts[3])
    elif encoder_name and 'nerf' in encoder_name.lower():
        # NeRF format: lr, frequency_num, ?, num_layers, hidden_dim
        # Example: 0.0100_64_0.1000000_2_512
        if len(numeric_parts) >= 5:
            result['lr'] = numeric_parts[0]
            result['frequency_num'] = int(numeric_parts[1])
            # numeric_parts[2] might be a NeRF-specific parameter, skip for now
            result['num_layers'] = int(numeric_parts[3])
            result['hidden_dim'] = int(numeric_parts[4])
        elif len(numeric_parts) >= 4:
            result['frequency_num'] = int(numeric_parts[0])
            result['num_layers'] = int(numeric_parts[2])
            result['hidden_dim'] = int(numeric_parts[3])
    elif encoder_name and 'sphere2vec' in encoder_name.lower():
        # Sphere2Vec format: lr, frequency_num, min_radius, num_layers, hidden_dim
        # Example: 0.0010_64_0.0010000_1_512 (no max_radius)
        if len(numeric_parts) >= 5:
            result['lr'] = numeric_parts[0]
            result['frequency_num'] = int(numeric_parts[1])
            result['min_radius'] = numeric_parts[2]
            result['num_layers'] = int(numeric_parts[3])
            result['hidden_dim'] = int(numeric_parts[4])
        elif len(numeric_parts) >= 4:
            # Sometimes lr might be missing
            result['frequency_num'] = int(numeric_parts[0])
            result['min_radius'] = numeric_parts[1]
            result['num_layers'] = int(numeric_parts[2])
            result['hidden_dim'] = int(numeric_parts[3])
    else:
        # Space2Vec and other encoders: lr, frequency_num, min_radius, max_radius, num_layers, hidden_dim
        # Example: 0.0100_128_0.1000000_360.000_1_512
        if len(numeric_parts) >= 6:
            result['lr'] = numeric_parts[0]
            result['frequency_num'] = int(numeric_parts[1])
            result['min_radius'] = numeric_parts[2]
            result['max_radius'] = numeric_parts[3]
            result['num_layers'] = int(numeric_parts[4])
            result['hidden_dim'] = int(numeric_parts[5])
        elif len(numeric_parts) >= 5:
            # Sometimes lr might be missing from filename
            result['frequency_num'] = int(numeric_parts[0])
            result['min_radius'] = numeric_parts[1]
            result['max_radius'] = numeric_parts[2]
            result['num_layers'] = int(numeric_parts[3])
            result['hidden_dim'] = int(numeric_parts[4])

    return result


def get_frequency_num_from_weights(checkpoint_path: str, encoder_name: str) -> Optional[int]:
    """
    Extract frequency_num (or legendre_poly_num for SIREN) directly from checkpoint weights.

    This is more reliable than parsing from filename for some encoders.

    Args:
        checkpoint_path: Path to checkpoint file
        encoder_name: Name of encoder (to determine layer structure)

    Returns:
        frequency_num (or legendre_poly_num) if found, None otherwise
    """
    try:
        import torch
        import math
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        weights = checkpoint.get('state_dict', {})

        # For NeRF and Space2Vec-grid, check ffn first layer input dimension
        # Position encoder output = coord_dim * frequency_num * factor
        # For NeRF: factor = 3 (x, sin, cos)
        # For Space2Vec: factor = 4 (sin_x, cos_x, sin_y, cos_y) for 2D
        # For SIREN: input_dim = legendre_poly_num^2
        for key in weights.keys():
            if 'spa_enc.ffn.layers.0.linear.weight' in key:
                input_dim = weights[key].shape[1]

                if 'siren' in encoder_name.lower() or 'spherical' in encoder_name.lower():
                    # Spherical Harmonics: input_dim = legendre_poly_num^2
                    legendre_poly_num = int(math.sqrt(input_dim))
                    return legendre_poly_num
                elif 'nerf' in encoder_name.lower():
                    # NeRF: input_dim = coord_dim * frequency_num * 3
                    # Assuming coord_dim = 2
                    frequency_num = input_dim // 6  # 2 * 3
                    return int(frequency_num)
                elif 'sphere2vec' in encoder_name.lower():
                    # Sphere2Vec: input_dim = frequency_num * 3
                    # Uses 3D spherical coordinates (x, y, z) directly
                    frequency_num = input_dim // 3
                    return int(frequency_num)
                elif 'space2vec' in encoder_name.lower():
                    # Space2Vec-grid: more complex, skip for now
                    pass

        return None
    except Exception:
        return None


def update_config_from_checkpoint(config: Dict, checkpoint_path: str) -> Dict:
    """
    Update config with hyperparameters parsed from checkpoint filename.

    This ensures the model architecture matches what was used during training.

    Args:
        config: Base configuration dictionary
        checkpoint_path: Path to checkpoint file

    Returns:
        Updated configuration with checkpoint-specific hyperparameters
    """
    parsed = parse_checkpoint_filename(checkpoint_path)

    # Update loc_encoder_params if they exist in parsed data
    if 'loc_encoder_params' in config and config['loc_encoder_params']:
        params = config['loc_encoder_params']

        # For frequency_num, prefer extracting from weights (more reliable)
        # EXCEPT for Sphere2Vec, where filename parsing works better
        encoder_name = config.get('loc_encoder_name', '')

        if 'sphere2vec' in encoder_name.lower():
            # For Sphere2Vec, use filename parsing directly (more reliable)
            if 'frequency_num' in parsed and 'frequency_num' in params:
                params['frequency_num'] = parsed['frequency_num']
        elif 'siren' in encoder_name.lower() or 'spherical' in encoder_name.lower():
            # For SIREN/Spherical Harmonics, extract legendre_poly_num from weights
            # (filename parsing doesn't work - the number in filename is NOT legendre_poly_num)
            legendre_from_weights = get_frequency_num_from_weights(checkpoint_path, encoder_name)
            if legendre_from_weights and 'legendre_poly_num' in params:
                params['legendre_poly_num'] = legendre_from_weights
        else:
            # For other encoders, prefer weights extraction
            freq_from_weights = get_frequency_num_from_weights(checkpoint_path, encoder_name)
            if freq_from_weights and 'frequency_num' in params:
                params['frequency_num'] = freq_from_weights
            elif 'frequency_num' in parsed and 'frequency_num' in params:
                params['frequency_num'] = parsed['frequency_num']
        if 'min_radius' in parsed and 'min_radius' in params:
            params['min_radius'] = parsed['min_radius']
        if 'max_radius' in parsed and 'max_radius' in params:
            params['max_radius'] = parsed['max_radius']
        if 'num_layers' in parsed and 'ffn_num_hidden_layers' in params:
            params['ffn_num_hidden_layers'] = parsed['num_layers']
        if 'hidden_dim' in parsed and 'ffn_hidden_dim' in params:
            params['ffn_hidden_dim'] = parsed['hidden_dim']
        if 'activation' in parsed and 'ffn_act' in params:
            params['ffn_act'] = parsed['activation']

    return config


# Test function
if __name__ == "__main__":
    test_filenames = [
        "model_birdsnap_ebird_meta_Space2Vec-grid_inception_v3_0.0100_128_0.1000000_360.000_1_512_leakyrelu.pth.tar",
        "model_nabirds_ebird_meta_Space2Vec-grid_inception_v3_0.0100_32_0.1000000_360.000_1_256_BATCH4096_leakyrelu.pth.tar",
        "model_birdsnap_ebird_meta_NeRF_inception_v3_0.0100_64_0.0200000_1_256_leakyrelu.pth.tar",
    ]

    for filename in test_filenames:
        print(f"\n{filename}")
        parsed = parse_checkpoint_filename(filename)
        print("Parsed parameters:")
        for key, value in parsed.items():
            print(f"  {key}: {value}")
