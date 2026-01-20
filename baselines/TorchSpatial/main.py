# Run this under the directory that contains TorchSpatial, not under TorchSpatial itself

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam

from sklearn.model_selection import train_test_split

from TorchSpatial.modules.trainer import train, forward_with_np_array
from TorchSpatial.modules.encoder_selector import get_loc_encoder
from TorchSpatial.modules.models import ThreeLayerMLP
import TorchSpatial.utils.datasets as data_import
import TorchSpatial.utils.eval_helper as eval_helper

from pathlib import Path
import numpy as np
import pandas as pd


def main():

    dataset = "mosaiks_elevation" # birdsnap and mosaiks_elevation work
    load_model = True
    train_model = True
    total_epochs = 3 # must always keep it correct for correct naming. 0 for new model, 3 for model that has been trained for 3 epochs, etc. The amount of epochs for which your model has already been trained. 
    epochs = 3 # only need to be kept correct if train_model is True. The amount of (additional) epochs for which you train your model by running this script

    # - import dataset
    settings = {"birdsnap":
                    {"params": {"dataset": "birdsnap", "meta_type": "orig_meta", "regress_dataset": []},
                     "task": "Classification",
                     "num_classes": 500,
                     "train_remove_invalid": True, # False can run (the original TorcHSpatial runs also use False), but the focus is "how would incorporating location data improve accuracy", and we want a clean comparison
                     "eval_remove_invalid": True},
                "mosaiks_elevation":
                    {"params": {"dataset": "mosaiks_elevation", "regress_dataset": ["mosaiks_elevation"]},
                     "task": "Regression",
                     "num_classes": 1,
                     "train_remove_invalid": False,
                     "eval_remove_invalid": False},
                }
    
    params = settings[dataset]["params"]
    task = settings[dataset]["task"]
    meta_type = params.get("meta_type", "")
    num_classes = settings[dataset]["num_classes"]
    train_remove_invalid = settings[dataset]["train_remove_invalid"]
    eval_remove_invalid = settings[dataset]["eval_remove_invalid"]

    eval_split = "test"

    device = "cpu"
    coord_dim = 2 #lonlat
    img_dim = loc_dim = embed_dim = 2048 # embedding dim
    batch_size = 32

    if task == "Classification":
        embed_dim = img_dim 
    elif task == "Regression": 
        embed_dim = img_dim + loc_dim
    
    # Allowed: Space2Vec-grid, Space2Vec-theory, xyz, NeRF, Sphere2Vec-sphereC, Sphere2Vec-sphereC+, Sphere2Vec-sphereM, Sphere2Vec-sphereM+, Sphere2Vec-dfs, rbf, rff, wrap, wrap_ffn, tile_ffn, Siren(SH)
    # For other required arguments, please refer to the docs (ex. rbf)
    # https://torchspatial.readthedocs.io/en/latest/2D%20Location%20Encoders/rbf.html
    loc_encoder_name = "Space2Vec-grid"
    loc_encoder_params = {"spa_embed_dim": loc_dim, "coord_dim":coord_dim, "device":device}

    all_data = data_import.load_dataset(params = params,
        eval_split = eval_split,
        train_remove_invalid = train_remove_invalid,
        eval_remove_invalid = eval_remove_invalid,
        load_cnn_predictions=True,
        load_cnn_features=True,
        load_cnn_features_train=True)

    if dataset == "birdsnap":
        img_tr = torch.Tensor(all_data["train_feats"]).long() # shape=(19133, 2048)
        loc_tr = torch.Tensor(all_data["train_locs"]).long() # shape=(19133, 2)
        y_tr = torch.Tensor(all_data["train_classes"]).long() # shape=(19133, )
    elif dataset == "mosaiks_elevation":
        img_tr = torch.Tensor(all_data["train_feats"]).float() # shape=(19924, 2048)
        loc_tr = torch.Tensor(all_data["train_locs"]).float() # shape=(19924, 2)
        y_tr = torch.Tensor(all_data["train_labels"]).float() # shape=(19924,)

    if dataset == "birdsnap":
        img_te = torch.Tensor(all_data["val_feats"]).long() # shape=(816, 2048)
        loc_te = torch.Tensor(all_data["val_locs"]).long() # shape=(816, 2)
        y_te = torch.Tensor(all_data["val_classes"]).long() # shape=(816, )
    elif dataset == "mosaiks_elevation":
        img_te = torch.Tensor(all_data["val_feats"]).float() # shape=(4981, 2048)
        loc_te = torch.Tensor(all_data["val_locs"]).float() # shape=(4981, 2)
        y_te = torch.Tensor(all_data["val_labels"]).float() # shape=(4981,)

    if task == "Regression":
        # Standardize img_tr and img_te (embeddings) to prevent exploding gradient
        img_tr_orig = img_tr
        img_te_orig = img_te
        img_mean = img_tr_orig.mean() # Use img_tr
        img_std  = img_tr_orig.std().clamp_min(1e-8)# Use img_tr
        img_tr = (img_tr_orig - img_mean) / img_std
        img_te = (img_te_orig - img_mean) / img_std

        # Standardize y_tr and y_te
        y_tr_orig = y_tr
        y_te_orig = y_te
        y_mean = y_tr_orig.mean() # Use y_tr
        y_std  = y_tr_orig.std().clamp_min(1e-8)# Use y_tr
        y_tr = (y_tr_orig - y_mean) / y_std
        y_te = (y_te_orig - y_mean) / y_std

    train_data_zip = list(zip(img_tr, loc_tr, y_tr))
    test_data_zip  = list(zip(img_te, loc_te, y_te))

    # - Dataloader (loads image embeddings)
    train_loader = DataLoader(train_data_zip, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_data_zip, batch_size=batch_size, shuffle=False)
    first_batch = next(iter(test_loader))
    img_b, loc_b, y_b = first_batch

    # - location encoder
    loc_encoder = get_loc_encoder(name = loc_encoder_name, overrides = loc_encoder_params).to(device) # "device": device is needed if you defined device = 'cpu' above and don't have cuda setup to prevent "AssertionError: Torch not compiled with CUDA enabled", because the default is device="cuda"

    # - model
    decoder = ThreeLayerMLP(input_dim = embed_dim, hidden_dim = 1024, category_count = num_classes).to(device)
    # - Criterion
    if task == "Classification":
        criterion = nn.CrossEntropyLoss()
    elif task == "Regression":
        criterion = nn.MSELoss()

    # - Optimizer
    optimizer = Adam(params = list(loc_encoder.parameters()) + list(decoder.parameters()), lr = 1e-3)

    # - Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, threshold=1e-4
    )

    if load_model:
        model_path = f"TorchSpatial/pre_trained_models/{loc_encoder_name.lower()}/model_{dataset}_{meta_type}_{loc_encoder_name}_{total_epochs}epochs.pth.tar"

        ckpt = torch.load(model_path, map_location=device)

        loc_encoder.load_state_dict(ckpt["loc_encoder"])
        decoder.load_state_dict(ckpt["decoder"])
        optimizer.load_state_dict(ckpt["optimizer"])
        total_epochs = ckpt["epoch"]
        scheduler.load_state_dict(ckpt["scheduler"])

        print(f"Checkpoint loaded from {model_path} at {total_epochs} epochs")

    loc_encoder.train()
    decoder.train()

    if train_model:
        # - train() 
        train(task = task,
                epochs = epochs, 
                batch_count_print_avg_loss = 30,
                loc_encoder = loc_encoder,
                dataloader = train_loader,
                decoder = decoder,
                criterion = criterion,
                optimizer = optimizer,
                scheduler = scheduler,
                device = device)
        total_epochs += epochs

    # - save model
    model_path = f"TorchSpatial/pre_trained_models/{loc_encoder_name.lower()}/model_{dataset}_{meta_type}_{loc_encoder_name}_{total_epochs}epochs.pth.tar"
    path = Path(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save({
        "epoch": total_epochs,
        "loc_encoder": loc_encoder.state_dict(),
        "decoder": decoder.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict()
    }, path)

    print(f"Model saved after {total_epochs} epochs as {model_path}")
    
    # - test
    loc_encoder.eval()
    decoder.eval()

    total = 0
    if task == "Classification":
        correct_top1 = 0
        correct_top3 = 0
        rr_sum = 0
    elif task == "Regression":
        sse = 0.0
        sst = 0.0
        sae = 0.0
        square_error = 0
        ys = []
        with torch.no_grad():
            for _, _, y_b in test_loader:
                ys.append(y_b.float().cpu())
        y_all = torch.cat(ys)
        ybar = y_all.mean().item()

    rows = []
    sample_id = 0


    with torch.no_grad():
        for img_b, loc_b, y_b in test_loader:

            img_b, loc_b, y_b = img_b.to(device), loc_b.to(device), y_b.to(device)

            img_embedding = img_b
            loc_embedding = forward_with_np_array(batch_data=loc_b, model=loc_encoder)

            if task == "Classification":
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
                top3_idx = logits.topk(3, dim=1).indices                    # [B, 3]
                correct_top3 += (top3_idx == y_b.unsqueeze(1)).any(dim=1).sum().item()
                hit_at_3 = (top3_idx == y_idx.unsqueeze(1)).any(dim=1)

                # MRR (full ranking over all classes)
                ranking = logits.argsort(dim=1, descending=True)             # [B, C]
                positions = ranking.argsort(dim=1)                           # [B, C] where positions[b, c] = rank index (0-based)
                true_pos0 = positions.gather(1, y_b.view(-1, 1)).squeeze(1)  # [B]
                rr_sum += (1.0 / (true_pos0.float() + 1.0)).sum().item()
                reciprocal_rank = 1.0 / (true_pos0.float() + 1.0)

                total += y_b.size(0)
                correct_top1 += (pred == y_b).sum().item()

                lon = loc_b[:,0]
                lat = loc_b[:,1]
                probas = nn.Softmax(dim = 1)(logits) 
                true_class_prob = probas.gather(1, y_idx.view(-1, 1)).squeeze(1)

                for i in range(B):
                    rows.append({
                        "Unnamed: 0": sample_id,
                        "lon": float(lon[i].item()),
                        "lat": float(lat[i].item()),
                        "true_class_prob": float(true_class_prob[i].item()),
                        "reciprocal_rank": float(reciprocal_rank[i].item()),
                        "hit@1": int(hit_at_1[i].item()), 
                        "hit@3": int(hit_at_3[i].item()),
                    })
                    sample_id += 1
            
            elif task == "Regression":
                loc_img_concat_embedding = torch.cat((loc_embedding, img_embedding), dim = 1)
                yhat = decoder(loc_img_concat_embedding).squeeze(-1)  # standardized during training, back during testing
    
                y_true = y_b.float()
                y_pred = yhat.float()

                B = y_b.size(0)

                err = y_true - y_pred
                sse += (err * err).sum().item()
                sae += err.abs().sum().item()
                sst += ((y_true - ybar) ** 2).sum().item()

                total += y_b.numel()
                epsilon = 1e-8
                lon = loc_b[:,0]
                lat = loc_b[:,1]
                predictions_raw = (y_pred * y_std) + y_mean
                labels_raw = (y_true * y_std) + y_mean
                relative_error = (predictions_raw - labels_raw) / (labels_raw + epsilon)

                for i in range(B):
                    rows.append({
                        "Unnamed: 0": sample_id,
                        "lon": float(lon[i].item()),
                        "lat": float(lat[i].item()),
                        "predictions": float(predictions_raw[i].item()),
                        "labels": float(labels_raw[i].item()),
                        "relative_error": float(relative_error[i].item()), 
                    })
                    sample_id += 1

    df = pd.DataFrame(rows)
    df.to_csv(f"TorchSpatial/eval_results/{task.lower()}/eval_{dataset}_{meta_type}_{eval_split}_{loc_encoder_name}_{total_epochs}epochs.csv", index=True)

    # Separate block because need to use total
    if task == "Classification":
        top1_acc = 100.0 * correct_top1 / total if total else 0.0
        top3_acc = 100.0 * correct_top3 / total if total else 0.0
        mrr = rr_sum / total if total else 0.0

        print(f"Top-1 Accuracy on {total} test images: {top1_acc:.2f}%")
        print(f"Top-3 Accuracy on {total} test images: {top3_acc:.2f}%")
        print(f"MRR on {total} test images: {mrr:.4f}")
    elif task == "Regression":
        rmse = (sse / total) ** 0.5
        mae = sae / total
        r2 = 1.0 - (sse / sst)

        rmse_raw = rmse * y_std
        mae_raw = mae * y_std
        print(f"r-square on {total} test images: {r2:.2f}")
        print(f"MAE of pred on {total} test images: {mae_raw:.2f}")
        print(f"RMSE of pred on {total} test images: {rmse_raw:.2f}")

    

if __name__ == "__main__":
    main()