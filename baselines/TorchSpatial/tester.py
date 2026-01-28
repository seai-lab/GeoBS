import torch
import torch.nn as nn

import numpy as np

# Some models like the included location encoders only supports list or np.ndarray
# Coerce datatype from torch.Tensor to np.ndarray briefly, then turn it back after processing
def forward_with_np_array(batch_data, model):
    loc_b = batch_data.detach().cpu().numpy() #loc_b = np.array(batch_data)
    loc_b = np.expand_dims(loc_b, axis=1) #loc_b = np.expand_dims(batch_data, axis=1)
    loc_embedding = torch.squeeze(model(coords = loc_b))
    return loc_embedding

def test(dataloader,
         loc_encoder,
         decoder,
         ssi_loss,
         ssi_partitioner,
         ssi_perf_transformer,
         sri_loss,
         sri_partitioner,
         sri_perf_transformer,
         scale_grid,
         distance_lag,
         split_number,
         device):

    total = 0.
    correct_top1 = 0.
    correct_top3 = 0.
    rr_sum = 0.
    ssis = []
    sri_sgs, sri_dls, sri_dss = [], [], []

    rows = []

    for idx_b, img_b, loc_b, y_b in dataloader:

        img_b, loc_b, y_b = img_b.to(device), loc_b.to(device), y_b.to(device)

        img_embedding = img_b
        if loc_encoder:
            loc_embedding = forward_with_np_array(batch_data = loc_b, model = loc_encoder)
        else:
            loc_embedding = torch.ones_like(img_embedding).float()
        loc_img_interaction_embedding = torch.mul(loc_embedding, img_embedding)

        logits = decoder(loc_img_interaction_embedding)

        B = y_b.size(0)

        if y_b.ndim == 2:
            y_idx = y_b.argmax(dim=1).long()
        else:
            y_idx = y_b.long()

        # Top-1
        pred = logits.argmax(dim=1)
        hit_at_1 = (pred == y_idx)

        # Top-3 accuracy
        top3_idx = logits.topk(3, dim=1).indices  # [B, 3]
        correct_top3 += (top3_idx == y_b.unsqueeze(1)).any(dim=1).sum().item()
        hit_at_3 = (top3_idx == y_idx.unsqueeze(1)).any(dim=1)

        # MRR (full ranking over all classes)
        ranking = logits.argsort(dim=1, descending=True)  # [B, C]
        positions = ranking.argsort(dim=1)  # [B, C] where positions[b, c] = rank index (0-based)
        true_pos0 = positions.gather(1, y_b.view(-1, 1)).squeeze(1)  # [B]
        rr_sum += (1.0 / (true_pos0.float() + 1.0)).sum().item()
        reciprocal_rank = 1.0 / (true_pos0.float() + 1.0)

        total += y_b.size(0)
        correct_top1 += (pred == y_b).sum().item()

        lon = loc_b[:, 0]
        lat = loc_b[:, 1]
        probas = nn.Softmax(dim=1)(logits)
        true_class_prob = probas.gather(1, y_idx.view(-1, 1)).squeeze(1)

        for i in range(B):
            idx = idx_b[i]

            neighborhood_idx = ssi_partitioner.get_neighborhood_idx(idx.item())
            tmp_ssi, ignore_ratio = None, None

            if neighborhood_idx.shape[0] >= 10:
                neighborhood_points = ssi_partitioner.get_neighborhood_points(idx.item())

                img_n, loc_n, y_n = (torch.stack([dataloader.dataset[i][1].to(device) for i in neighborhood_idx]),
                                     torch.stack([dataloader.dataset[i][2].to(device) for i in neighborhood_idx]),
                                     torch.stack([dataloader.dataset[i][3].to(device) for i in neighborhood_idx]))

                img_embedding = img_n
                if loc_encoder:
                    loc_embedding = forward_with_np_array(batch_data = loc_n, model = loc_encoder)
                else:
                    loc_embedding = torch.ones_like(img_embedding).float()
                loc_img_interaction_embedding = torch.mul(loc_embedding, img_embedding)
                
                logits = decoder(loc_img_interaction_embedding)

                neighborhood_values = ssi_perf_transformer(logits, y_n)

                tmp_ssi, ignore_ratio = ssi_loss(neighborhood_points, neighborhood_values)
                ignore_ratio = float(ignore_ratio)

                if tmp_ssi is not None:
                    tmp_ssi = float(tmp_ssi[0].item())
                    ssis.append(tmp_ssi)

            tmp_sri_sg, tmp_sri_dl, tmp_sri_ds = None, None, None
            tmp_sri_sgs, tmp_sri_dls, tmp_sri_dss = [], [], []

            neighborhood_idx, _, _ = sri_partitioner.get_neighborhood_idx(idx.item())

            if neighborhood_idx.shape[0] > 50:

                img_n, loc_n, y_n = (torch.stack([dataloader.dataset[i][1].to(device) for i in neighborhood_idx]),
                                    torch.stack([dataloader.dataset[i][2].to(device) for i in neighborhood_idx]),
                                    torch.stack([dataloader.dataset[i][3].to(device) for i in neighborhood_idx]))

                img_embedding = img_n
                if loc_encoder:
                    loc_embedding = forward_with_np_array(batch_data=loc_n, model=loc_encoder)
                else:
                    loc_embedding = torch.ones_like(img_embedding).float()
                loc_img_interaction_embedding = torch.mul(loc_embedding, img_embedding)

                logits = decoder(loc_img_interaction_embedding)

                neighborhood_values = sri_perf_transformer(logits, y_n)

                print("Neighborhood values", neighborhood_values)

                # Scale Grid SRI
                partition_idx_list, neighborhood_idx = sri_partitioner.get_scale_grid_idx(idx.item(), scale=scale_grid)
                for partition_idx in partition_idx_list:
                    partition_values = sri_perf_transformer(logits[partition_idx], y_n[partition_idx])
                    print("Scale grid values", partition_values)
                    tmp_sri_sgs.append(sri_loss(partition_values, neighborhood_values).item())

                if len(tmp_sri_sgs) > 0:
                    tmp_sri_sg = np.sum(tmp_sri_sgs)
                    sri_sgs.append(tmp_sri_sg)

                # Distance Lag SRI
                partition_idx_list, neighborhood_idx = sri_partitioner.get_distance_lag_idx(idx.item(), lag=distance_lag)
                for partition_idx in partition_idx_list:
                    partition_values = sri_perf_transformer(logits[partition_idx], y_n[partition_idx])
                    print("Distance lag values", partition_values)
                    tmp_sri_dls.append(sri_loss(partition_values, neighborhood_values).item())

                if len(tmp_sri_dls) > 0:
                    tmp_sri_dl = np.sum(tmp_sri_dls)
                    sri_dls.append(tmp_sri_dl)

                # Direction Sector SRI
                partition_idx_list, neighborhood_idx = sri_partitioner.get_direction_sector_idx(idx.item(), n_splits=split_number)
                for partition_idx in partition_idx_list:
                    partition_values = sri_perf_transformer(logits[partition_idx], y_n[partition_idx])
                    print("Direction sector values", partition_values)
                    tmp_sri_dss.append(sri_loss(partition_values, neighborhood_values).item())

                if len(tmp_sri_dss) > 0:
                    tmp_sri_ds = np.sum(tmp_sri_dss)
                    sri_dss.append(tmp_sri_ds)

            rows.append({
                "lon": float(lon[i].item()),
                "lat": float(lat[i].item()),
                "true_class_prob": float(true_class_prob[i].item()),
                "reciprocal_rank": float(reciprocal_rank[i].item()),
                "hit@1": int(hit_at_1[i].item()),
                "hit@3": int(hit_at_3[i].item()),
                "ssi": tmp_ssi,
                # "ignore_ratio": ignore_ratio
                "sri_sg": tmp_sri_sg,
                "sri_dl": tmp_sri_dl,
                "sri_ds": tmp_sri_ds
            })

    # Separate block because need to use total
    top1_acc = 100.0 * correct_top1 / total if total else 0.0
    top3_acc = 100.0 * correct_top3 / total if total else 0.0
    mrr = rr_sum / total if total else 0.0
    ssi = np.mean(ssis) if len(ssis) > 0 else 0.0
    sri_sg = np.mean(sri_sgs) if len(sri_sgs) > 0 else 0.0
    sri_dl = np.mean(sri_dls) if len(sri_dls) > 0 else 0.0
    sri_ds = np.mean(sri_dss) if len(sri_dss) > 0 else 0.0

    print(f"Top-1 Accuracy on {total} test images: {top1_acc:.2f}%")
    print(f"Top-3 Accuracy on {total} test images: {top3_acc:.2f}%")
    print(f"MRR on {total} test images: {mrr:.4f}")
    print(f"SSI score on {len(ssis)} valid test neighborhoods: {ssi:.4f}")
    print(f"SRI SG score on {len(sri_sgs)} valid test neighborhoods: {sri_sg:.4f}")
    print(f"SSI DL score on {len(sri_dls)} valid test neighborhoods: {sri_dl:.4f}")
    print(f"SSI DS score on {len(sri_dss)} valid test neighborhoods: {sri_ds:.4f}")

    return rows