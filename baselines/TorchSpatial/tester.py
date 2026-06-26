"""
Unified tester for all location encoder types.

This replaces the redundant tester_space2vec_grid.py, tester_no_prior.py, and tester.py
which were nearly identical copies (~678 lines total).

Usage:
    from TorchSpatial.tester_unified import test

    # With location encoder (Space2Vec-grid, NeRF, xyz, etc.)
    rows = test(dataloader, loc_encoder=model)

    # Without location encoder (no_prior baseline)
    rows = test(dataloader, loc_encoder=None)

The unified approach follows the original TorchSpatial architecture which uses
a single eval_helper.py with parameterized prior_type, rather than separate files.
"""

import torch
import torch.nn as nn
import numpy as np


def forward_with_np_array(batch_data, model):
    """
    Helper function for models that only support list or np.ndarray inputs.
    Coerce datatype from torch.Tensor to np.ndarray briefly, then convert back.

    Note: This is kept for backward compatibility but may not be needed for most encoders.
    """
    loc_b = batch_data.detach().cpu().numpy()
    loc_b = np.expand_dims(loc_b, axis=1)
    loc_embedding = torch.squeeze(model(coords=loc_b))
    return loc_embedding


def test(dataloader, loc_encoder=None):
    """
    Unified test function for all location encoder types.

    Args:
        dataloader: PyTorch DataLoader yielding (idx, img_pred, loc, y_true) batches
        loc_encoder: Location encoder model (Space2Vec-grid, xyz, NeRF, etc.) or None for no_prior baseline

    Returns:
        List of dicts containing per-sample results:
            - lon, lat: location coordinates
            - true_class_prob: predicted probability for the true class
            - reciprocal_rank: 1 / rank of true class
            - hit@1: 1 if top-1 prediction is correct, else 0
            - hit@3: 1 if true class in top-3, else 0

    Notes:
        - All spatial encoders (Space2Vec-*, xyz, NeRF, Sphere2Vec-*, rbf, rff, wrap, tile_ffn)
          share the same forward interface and can be used interchangeably
        - LocationEncoder.forward() internally applies sigmoid (see models.py line 480),
          so we do NOT apply sigmoid again here (that was the double sigmoid bug)
        - For no_prior baseline, simply pass loc_encoder=None
    """

    total = 0
    correct_top1 = 0
    correct_top3 = 0
    rr_sum = 0.0
    rows = []

    for idx_b, img_b, loc_b, y_b in dataloader:
        # CNN predictions (from pre-computed features)
        class_probas_based_on_image = img_b

        # Location prior (if encoder provided)
        if loc_encoder:
            # Mark rows containing any NaN in location as unusable
            valid_loc_mask = ~torch.isnan(loc_b).any(dim=1)
            loc_embedding = torch.ones_like(class_probas_based_on_image).float()

            if valid_loc_mask.any():
                # Forward pass through location encoder
                # NOTE: All spatial encoders implement the same interface:
                #   input: (batch_size, 2) coordinates
                #   output: (batch_size, num_classes) probabilities (post-sigmoid)
                loc_embedding[valid_loc_mask] = loc_encoder(loc_b[valid_loc_mask])

            # CRITICAL FIX (2026-06-24):
            # LocationEncoder.forward() already applies sigmoid internally (models.py line 480).
            # Do NOT apply sigmoid again here - that causes "double sigmoid" which compresses
            # all probabilities to ~0.5, destroying location prior's discriminative power.
            # This bug caused 8% accuracy drop (80.24% → 72%).
            class_probas_based_on_loc = loc_embedding
        else:
            # No prior baseline: uniform location prior (all classes equally likely everywhere)
            class_probas_based_on_loc = torch.ones_like(class_probas_based_on_image).float()

        # Combine CNN and location predictions (element-wise multiplication)
        class_probas = class_probas_based_on_loc * class_probas_based_on_image

        B = y_b.size(0)
        y_idx = y_b.long()

        # Compute ranks: sorted indices descending by predicted probability
        sorted_idx = torch.argsort(class_probas, dim=1, descending=True)   # [B, C]
        positions = torch.argsort(sorted_idx, dim=1)                       # [B, C]
        true_rank = positions.gather(1, y_idx.view(-1, 1)).squeeze(1) + 1  # [B], 1-based

        # Metrics
        hit_at_1 = (true_rank <= 1)
        hit_at_3 = (true_rank <= 3)
        reciprocal_rank = 1.0 / true_rank.float()
        true_class_prob = class_probas.gather(1, y_idx.view(-1, 1)).squeeze(1)

        # Accumulate batch statistics
        correct_top1 += hit_at_1.sum().item()
        correct_top3 += hit_at_3.sum().item()
        rr_sum += reciprocal_rank.sum().item()
        total += B

        # Extract location coordinates
        lon = loc_b[:, 0]
        lat = loc_b[:, 1]

        # Store per-sample results
        for i in range(B):
            rows.append({
                "lon": float(lon[i].item()),
                "lat": float(lat[i].item()),
                "true_class_prob": float(true_class_prob[i].item()),
                "reciprocal_rank": float(reciprocal_rank[i].item()),
                "hit@1": int(hit_at_1[i].item()),
                "hit@3": int(hit_at_3[i].item()),
            })

    # Print summary statistics
    top1_acc = 100.0 * correct_top1 / total if total else 0.0
    top3_acc = 100.0 * correct_top3 / total if total else 0.0
    mrr = rr_sum / total if total else 0.0

    encoder_type = "with location encoder" if loc_encoder else "no_prior baseline"
    print(f"\n=== Evaluation Results ({encoder_type}) ===")
    print(f"Total samples: {total}")
    print(f"Top-1 Accuracy: {top1_acc:.2f}%")
    print(f"Top-3 Accuracy: {top3_acc:.2f}%")
    print(f"MRR: {mrr:.4f}")

    return rows
