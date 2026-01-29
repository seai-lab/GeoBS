# Run this under the directory that contains TorchSpatial, not under TorchSpatial itself

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam

from sklearn.model_selection import train_test_split

from TorchSpatial.trainer import train, train_sri_debias
from TorchSpatial.tester import test
from TorchSpatial.modules.encoder_selector import get_loc_encoder
from TorchSpatial.modules.models import ThreeLayerMLP
import TorchSpatial.utils.datasets as data_import
import TorchSpatial.utils.eval_helper as eval_helper

from gbsloss import SSIPartitioner, BinaryPerformanceTransformer, SSILoss, SRIPartitioner, SoftHistogramPerformanceTransformer, SRILoss

from pathlib import Path
import numpy as np
import pandas as pd

import torch
import numpy as np

import json

import warnings

def main():

    # - import configs
    with open("configs.json", "r") as f:
        settings = json.load(f)

    dataset = settings["dataset"]
    eval_split = settings["eval_split"]
    load_model = settings["load_model"]
    debias_lambda = settings["debias_lambda"]

    ssi_radius = settings["ssi_radius"]
    sri_radius = settings["sri_radius"]
    partition_mode = settings["sri_partition_mode"]
    scale_grid = settings["sri_scale_grid"]
    distance_lag = settings["sri_distance_lag"]
    split_number = settings["sri_split_number"]

    trained_epochs = settings["trained_epochs"]
    debiased_epochs = settings["debiased_epochs"]
    epochs_to_train = settings["epochs_to_train"]
    epochs_to_debias = settings["epochs_to_debias"]

    loc_encoder_name = settings["loc_encoder_name"]
    loc_encoder_params = settings["loc_encoder_params"]
    batch_size = settings["batch_size"]
    batch_count_print_avg_loss = settings["batch_count_print_avg_loss"]
    decoder_hidden_dim = settings["decoder_hidden_dim"]
    activation_func = settings["activation_func"]

    optimizer_lr = settings["optimizer_lr"]
    scheduler_threshold = settings["scheduler_threshold"]

    partition_k = settings["partition_k"]
    BinaryPerformanceTransformer_thres = settings["BinaryPerformanceTransformer_thres"]
    SoftHistogramPerformanceTransformer_bins = settings["SoftHistogramPerformanceTransformer_bins"]

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
    
    embed_dim = img_dim 
    
    
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

    img_tr = torch.Tensor(all_data["train_feats"]).long() # shape=(N, 2048)
    loc_tr = torch.Tensor(all_data["train_locs"]).long() # shape=(N, 2)
    y_tr = torch.Tensor(all_data["train_classes"]).long() # shape=(N, )

    if loc_encoder_name == "rbf":
        loc_encoder_params["train_locs"] = all_data["train_locs"]
    
    img_te = torch.Tensor(all_data["val_feats"]).long() # shape=(N, 2048)
    loc_te = torch.Tensor(all_data["val_locs"]).long() # shape=(N, 2)
    y_te = torch.Tensor(all_data["val_classes"]).long() # shape=(N, )

    idx_tr = np.arange(img_tr.shape[0])
    idx_te = np.arange(img_te.shape[0])

    train_data_zip = list(zip(idx_tr, img_tr, loc_tr, y_tr))
    test_data_zip = list(zip(idx_te, img_te, loc_te, y_te))

    print("Check the radian of input data!", loc_tr[0])

    # - Dataloader (loads image embeddings)
    train_loader = DataLoader(train_data_zip, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_data_zip, batch_size=batch_size, shuffle=False)

    # - location encoder
    if loc_encoder_name != "no_prior":
        loc_encoder = get_loc_encoder(name = loc_encoder_name, overrides = loc_encoder_params).to(device) # "device": device is needed if you defined device = 'cpu' above and don't have cuda setup to prevent "AssertionError: Torch not compiled with CUDA enabled", because the default is device="cuda"
    else:
        loc_encoder = None

    # - model
    decoder = ThreeLayerMLP(input_dim = embed_dim, hidden_dim = decoder_hidden_dim, category_count = num_classes, activation_func = activation_func).to(device)
    # - Criterion
    criterion = nn.CrossEntropyLoss()

    # - Optimizer
    if loc_encoder:
        optimizer = Adam(params = list(loc_encoder.parameters()) + list(decoder.parameters()), lr = optimizer_lr)
    else:
        optimizer = Adam(params = list(decoder.parameters()), lr = optimizer_lr)

    # - Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, threshold=scheduler_threshold
    )

    epochs_order = []

    if load_model:
        model_path = f"TorchSpatial/pre_trained_models/{loc_encoder_name.lower()}/model_{dataset}_{meta_type}_{loc_encoder_name}_trained{trained_epochs}_debiased{debiased_epochs}.pth.tar"

        ckpt = torch.load(model_path, map_location=device)
        if loc_encoder:
            loc_encoder.load_state_dict(ckpt["loc_encoder"])
        decoder.load_state_dict(ckpt["decoder"])
        optimizer.load_state_dict(ckpt["optimizer"])
        trained_epochs = ckpt["trained_epochs"]
        debiased_epochs = ckpt["debiased_epochs"]
        epochs_order = ckpt["epochs_order"]
        scheduler.load_state_dict(ckpt["scheduler"])

        print(f"Checkpoint loaded from {model_path}; trained for {trained_epochs} epochs, debiased for {debiased_epochs} epochs, in the order of {epochs_order}")

    if loc_encoder:
        loc_encoder.train()
    decoder.train()

    ### Initialize gbs loss meta
    ssi_loss = SSILoss()
    sri_loss = SRILoss()

    lats, lons = np.radians(loc_tr[:, 1]), np.radians(loc_tr[:, 0])
    # ssi_partitioner = SSIPartitioner(np.array([lats, lons]).T, k=partition_k, radius=ssi_radius)
    # ssi_perf_transformer = BinaryPerformanceTransformer(thres=BinaryPerformanceTransformer_thres)
    sri_partitioner = SRIPartitioner(np.array([lats, lons]).T, k=partition_k, radius=sri_radius)
    sri_perf_transformer = SoftHistogramPerformanceTransformer(bins=SoftHistogramPerformanceTransformer_bins)

    train(epochs=epochs_to_train,
        batch_count_print_avg_loss=batch_count_print_avg_loss,
        loc_encoder=loc_encoder,
        dataloader=train_loader,
        decoder=decoder,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device)
    
    if epochs_to_train:
        trained_epochs += epochs_to_train
        epochs_order.append(("train", epochs_to_train))

    # - debias
    train_sri_debias(epochs = epochs_to_debias,
        batch_count_print_avg_loss = batch_count_print_avg_loss,
        loc_encoder = loc_encoder,
        dataloader = train_loader,
        decoder = decoder,
        criterion = criterion,
        debias_loss = sri_loss,
        debias_lambda = debias_lambda,
        partitioner = sri_partitioner,
        partition_mode = partition_mode,
        scale_grid = scale_grid,
        distance_lag = distance_lag,
        split_number = split_number,
        perf_transformer = sri_perf_transformer,
        optimizer = optimizer,
        scheduler = scheduler,
        device = device)
        
    if epochs_to_debias:
        debiased_epochs += epochs_to_debias
        epochs_order.append(("debias", epochs_to_debias))

    # - save model
    model_path = f"TorchSpatial/pre_trained_models/{loc_encoder_name.lower()}/model_{dataset}_{meta_type}_{loc_encoder_name}_trained{trained_epochs}_debiased{debiased_epochs}.pth.tar"
    path = Path(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if loc_encoder:
        torch.save({
            "trained_epochs": trained_epochs,
            "debiased_epochs": debiased_epochs,
            "epochs_order": epochs_order,
            "loc_encoder": loc_encoder.state_dict(),
            "decoder": decoder.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict()
        }, path)
    else:
        torch.save({
            "trained_epochs": trained_epochs,
            "debiased_epochs": debiased_epochs,
            "epochs_order": epochs_order,
            "decoder": decoder.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict()
        }, path)

    print(f"Model saved as {model_path}; in total, trained for {trained_epochs} epochs, debiased for {debiased_epochs} epochs, in the order of {epochs_order}")
    
    # - test
    if loc_encoder:
        loc_encoder.eval()
    decoder.eval()

    with torch.no_grad():

        lats, lons = np.radians(loc_te[:, 1]), np.radians(loc_te[:, 0])
        test_ssi_partitioner = SSIPartitioner(np.array([lats, lons]).T, k=partition_k, radius=ssi_radius)
        test_ssi_perf_transformer = BinaryPerformanceTransformer(thres=BinaryPerformanceTransformer_thres)
        test_sri_partitioner = SRIPartitioner(np.array([lats, lons]).T, k=partition_k, radius=sri_radius)
        test_sri_perf_transformer = SoftHistogramPerformanceTransformer(bins=SoftHistogramPerformanceTransformer_bins)

        rows = test(test_loader,
                    loc_encoder,
                    decoder,
                    ssi_loss,
                    test_ssi_partitioner,
                    test_ssi_perf_transformer,
                    sri_loss,
                    test_sri_partitioner,
                    test_sri_perf_transformer,
                    scale_grid,
                    distance_lag,
                    split_number,
                    device)

    df = pd.DataFrame(rows)
    df.to_csv(f"TorchSpatial/eval_results/eval_{dataset}_{meta_type}_{eval_split}_{loc_encoder_name}_trained-{trained_epochs}_debiased-{debiased_epochs}.csv", index=True)

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()