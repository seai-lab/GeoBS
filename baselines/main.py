# Run this under the directory that contains TorchSpatial, not under TorchSpatial itself

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam

from sklearn.model_selection import train_test_split

from TorchSpatial.modules.trainer import train, train_debias , forward_with_np_array
from TorchSpatial.modules.encoder_selector import get_loc_encoder
from TorchSpatial.modules.models import ThreeLayerMLP
import TorchSpatial.utils.datasets as data_import
import TorchSpatial.utils.eval_helper as eval_helper

from gbsloss import SSIPartitioner, BinaryPerformanceTransformer, LogOddsPerformanceTransformer, SSILoss

from pathlib import Path
import numpy as np
import pandas as pd

import torch
import numpy as np

import json

def main():

    # - import configs
    with open("configs.json", "r") as f:
        settings = json.load(f)

    dataset = settings["dataset"]
    eval_split = settings["eval_split"]
    load_model = settings["load_model"]
    train_model = settings["train_model"]
    debias = settings["debias"]
    debias_radius = settings["debias_radius"]
    debias_lambda = settings["debias_lambda"]

    trained_epochs = settings["trained_epochs"]
    debiased_epochs = settings["debiased_epochs"]
    nominal_epochs_to_train = settings["nominal_epochs_to_train"] # "nominal" because: either 1. train for 2 * nominal_epochs_to_train, or 2. train for nominal_epochs_to_train and debias for nominal_epochs_to_train

    loc_encoder_name = settings["loc_encoder_name"]
    loc_encoder_params = settings["loc_encoder_params"]
    batch_size = settings["batch_size"]
    batch_count_print_avg_loss = settings["batch_count_print_avg_loss"]
    decoder_hidden_dim = settings["decoder_hidden_dim"]
    activation_func = settings["activation_func"]

    optimizer_lr = settings["optimizer_lr"]
    scheduler_threshold = settings["scheduler_threshold"]

    SSIPartitioner_k = settings["SSIPartitioner_k"]
    BinaryPerformanceTransformer_thres = settings["BinaryPerformanceTransformer_thres"]

    params = settings[dataset]["params"]
    task = settings[dataset]["task"]
    meta_type = params.get("meta_type", "")
    img_dim = settings[dataset]["img_dim"]
    coord_dim = settings[dataset]["coord_dim"] #lonlat
    num_classes = settings[dataset]["num_classes"]
    train_remove_invalid = settings[dataset]["train_remove_invalid"]
    eval_remove_invalid = settings[dataset]["eval_remove_invalid"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loc_dim = img_dim
    if task == "Classification":
        embed_dim = img_dim 
    elif task == "Regression": 
        embed_dim = img_dim + loc_dim
    
    # Allowed: Space2Vec-grid, Space2Vec-theory, xyz, NeRF, Sphere2Vec-sphereC, Sphere2Vec-sphereC+, Sphere2Vec-sphereM, Sphere2Vec-sphereM+, Sphere2Vec-dfs, rbf, rff, wrap, wrap_ffn, tile_ffn, Siren(SH)
    # For other required arguments, please refer to the docs (ex. rbf)
    # https://torchspatial.readthedocs.io/en/latest/2D%20Location%20Encoders/rbf.html
    loc_encoder_params["device"] = device

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

    idx_tr = np.arange(img_tr.shape[0])
    idx_te = np.arange(img_te.shape[0])

    train_data_zip = list(zip(idx_tr, img_tr, loc_tr, y_tr))
    test_data_zip = list(zip(idx_te, img_te, loc_te, y_te))

    print("Check the radian of input data!", loc_tr[0])

    # - Dataloader (loads image embeddings)
    train_loader = DataLoader(train_data_zip, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_data_zip, batch_size=batch_size, shuffle=False)
    # first_batch = next(iter(test_loader))
    # img_b, loc_b, y_b = first_batch

    # - location encoder
    loc_encoder = get_loc_encoder(name = loc_encoder_name, overrides = loc_encoder_params).to(device) # "device": device is needed if you defined device = 'cpu' above and don't have cuda setup to prevent "AssertionError: Torch not compiled with CUDA enabled", because the default is device="cuda"

    # - model
    decoder = ThreeLayerMLP(input_dim = embed_dim, hidden_dim = decoder_hidden_dim, category_count = num_classes, activation_func = activation_func).to(device)
    # - Criterion
    if task == "Classification":
        criterion = nn.CrossEntropyLoss()
    elif task == "Regression":
        criterion = nn.MSELoss()

    # - Optimizer
    optimizer = Adam(params = list(loc_encoder.parameters()) + list(decoder.parameters()), lr = optimizer_lr)

    # - Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, threshold=scheduler_threshold
    )

    epochs_order = []

    if load_model:
        model_path = f"TorchSpatial/pre_trained_models/{loc_encoder_name.lower()}/model_{dataset}_{meta_type}_{loc_encoder_name}_trained{trained_epochs}_debiased{debiased_epochs}.pth.tar"

        ckpt = torch.load(model_path, map_location=device)

        loc_encoder.load_state_dict(ckpt["loc_encoder"])
        decoder.load_state_dict(ckpt["decoder"])
        optimizer.load_state_dict(ckpt["optimizer"])
        trained_epochs = ckpt["trained_epochs"]
        debiased_epochs = ckpt["debiased_epochs"]
        epochs_order = ckpt["epochs_order"]
        scheduler.load_state_dict(ckpt["scheduler"])

        print(f"Checkpoint loaded from {model_path}; trained for {trained_epochs} epochs, debiased for {debiased_epochs} epochs, in the order of {epochs_order}")

    loc_encoder.train()
    decoder.train()

    ### Initialize gbs loss meta
    debias_loss = SSILoss()

    lats, lons = np.radians(loc_tr[:,1]), np.radians(loc_tr[:,0])
    partitioner = SSIPartitioner(np.array([lats, lons]).T, k=SSIPartitioner_k, radius=debias_radius)

    # perf_transformer = LogOddsPerformanceTransformer(num_classes, scale=10)
    perf_transformer = BinaryPerformanceTransformer(thres=BinaryPerformanceTransformer_thres)

    if train_model:
        # - train()
        train(task=task,
              epochs=nominal_epochs_to_train,
              batch_count_print_avg_loss=batch_count_print_avg_loss,
              loc_encoder=loc_encoder,
              dataloader=train_loader,
              decoder=decoder,
              criterion=criterion,
              optimizer=optimizer,
              scheduler=scheduler,
              device=device)

        trained_epochs += nominal_epochs_to_train

        # - debias train
        if debias:
            train_debias(task = task,
                epochs = nominal_epochs_to_train, 
                batch_count_print_avg_loss = batch_count_print_avg_loss,
                loc_encoder = loc_encoder,
                dataloader = train_loader,
                decoder = decoder,
                criterion = criterion,
                debias_loss = debias_loss,
                debias_lambda = debias_lambda,
                partitioner = partitioner,
                perf_transformer = perf_transformer,
                optimizer = optimizer,
                scheduler = scheduler,
                device = device)
            
            debiased_epochs += nominal_epochs_to_train
            epochs_order.extend([("train", nominal_epochs_to_train), ("debias", nominal_epochs_to_train)])
        
        else:
            train(task=task,
                  epochs=nominal_epochs_to_train,
                  batch_count_print_avg_loss=batch_count_print_avg_loss,
                  loc_encoder=loc_encoder,
                  dataloader=train_loader,
                  decoder=decoder,
                  criterion=criterion,
                  optimizer=optimizer,
                  scheduler=scheduler,
                  device=device)
            
            trained_epochs += nominal_epochs_to_train
            epochs_order.append(("train", 2 * nominal_epochs_to_train))

    # - save model
    model_path = f"TorchSpatial/pre_trained_models/{loc_encoder_name.lower()}/model_{dataset}_{meta_type}_{loc_encoder_name}_trained{trained_epochs}_debiased{debiased_epochs}.pth.tar"
    path = Path(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save({
        "trained_epochs": trained_epochs,
        "debiased_epochs": debiased_epochs,
        "epochs_order": epochs_order,
        "loc_encoder": loc_encoder.state_dict(),
        "decoder": decoder.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict()
    }, path)

    print(f"Model saved as {model_path}; in total, trained for {trained_epochs} epochs, debiased for {debiased_epochs} epochs, in the order of {epochs_order}")
    
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
                ys.append(y_b.float().to(device))
        y_all = torch.cat(ys)
        ybar = y_all.mean().item()

    rows = []
    sample_id = 0

    total_ssi = 0.

    with torch.no_grad():

        lats, lons = np.radians(loc_te[:, 1]), np.radians(loc_te[:, 0])
        partitioner = SSIPartitioner(np.array([lats, lons]).T, k=100, radius=debias_radius)
        perf_transformer = BinaryPerformanceTransformer(thres=0.9)

        for idx_b, img_b, loc_b, y_b in test_loader:

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

                ### SSI evaluation
                for idx in idx_b:
                    neighborhood_idx = partitioner.get_neighborhood_idx(idx.item())
                    if neighborhood_idx.shape[0] < 10:
                        continue

                    neighborhood_points = partitioner.get_neighborhood_points(idx.item())

                    img_n, loc_n, y_n = (torch.stack([test_loader.dataset[i][1].to(device) for i in neighborhood_idx]),
                                         torch.stack([test_loader.dataset[i][2].to(device) for i in neighborhood_idx]),
                                         torch.stack([test_loader.dataset[i][3].to(device) for i in neighborhood_idx]))

                    img_embedding = img_n
                    loc_embedding = forward_with_np_array(batch_data=loc_n, model=loc_encoder)

                    loc_img_interaction_embedding = torch.mul(loc_embedding, img_embedding)
                    logits = decoder(loc_img_interaction_embedding)

                    neighborhood_values = perf_transformer(logits, y_n)

                    total_ssi += debias_loss(neighborhood_points, neighborhood_values)[0].item()
            
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
    df.to_csv(f"TorchSpatial/eval_results/{task.lower()}/eval_{dataset}_{meta_type}_{eval_split}_{loc_encoder_name}__trained{trained_epochs}_debiased{debiased_epochs}.csv", index=True)

    # Separate block because need to use total
    if task == "Classification":
        top1_acc = 100.0 * correct_top1 / total if total else 0.0
        top3_acc = 100.0 * correct_top3 / total if total else 0.0
        mrr = rr_sum / total if total else 0.0
        ssi = total_ssi / total if total else 0.0

        print(f"Top-1 Accuracy on {total} test images: {top1_acc:.2f}%")
        print(f"Top-3 Accuracy on {total} test images: {top3_acc:.2f}%")
        print(f"MRR on {total} test images: {mrr:.4f}")
        print(f"SSI score on {total} test images: {ssi:.4f}")
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