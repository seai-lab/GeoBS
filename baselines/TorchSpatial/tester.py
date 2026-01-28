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
         debias_loss,
         partitioner,
         perf_transformer,
         device):

    total = 0.
    correct_top1 = 0.
    correct_top3 = 0.
    rr_sum = 0.
    gbses = []

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

            neighborhood_idx = partitioner.get_neighborhood_idx(idx.item())
            tmp_gbs, ignore_ratio = None, None

            if neighborhood_idx.shape[0] >= 10:
                neighborhood_points = partitioner.get_neighborhood_points(idx.item())

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

                neighborhood_values = perf_transformer(logits, y_n)

                tmp_gbs, ignore_ratio = debias_loss(neighborhood_points, neighborhood_values)
                ignore_ratio = float(ignore_ratio)

                if tmp_gbs is not None:
                    tmp_gbs = float(tmp_gbs[0].item())
                    gbses.append(tmp_gbs)

            rows.append({
                "lon": float(lon[i].item()),
                "lat": float(lat[i].item()),
                "true_class_prob": float(true_class_prob[i].item()),
                "reciprocal_rank": float(reciprocal_rank[i].item()),
                "hit@1": int(hit_at_1[i].item()),
                "hit@3": int(hit_at_3[i].item()),
                "gbs": tmp_gbs,
                "ignore_ratio": ignore_ratio
            })

    # Separate block because need to use total
    top1_acc = 100.0 * correct_top1 / total if total else 0.0
    top3_acc = 100.0 * correct_top3 / total if total else 0.0
    mrr = rr_sum / total if total else 0.0
    gbs = np.mean(gbses) if len(gbses) > 0 else 0.0

    print(f"Top-1 Accuracy on {total} test images: {top1_acc:.2f}%")
    print(f"Top-3 Accuracy on {total} test images: {top3_acc:.2f}%")
    print(f"MRR on {total} test images: {mrr:.4f}")
    print(f"GBS score on {len(gbses)} valid test neighborhoods: {gbs:.4f}")

    return rows