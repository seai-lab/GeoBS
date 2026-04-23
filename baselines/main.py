# Run this under the directory that contains TorchSpatial, not under TorchSpatial itself

import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam

from sklearn.model_selection import train_test_split

from TorchSpatial.trainer import train, train_ssi_debias
from TorchSpatial.tester import test
from TorchSpatial.modules.encoder_selector import get_loc_encoder
from TorchSpatial.modules.models import ThreeLayerMLP
import TorchSpatial.utils.datasets as data_import
import TorchSpatial.utils.eval_helper as eval_helper
from TorchSpatial.utils.loss_registry import get_loss

from gbsloss import SSIPartitioner, BinaryPerformanceTransformer, SSILoss, SRIPartitioner, SoftHistogramPerformanceTransformer, SRILoss

from pathlib import Path
import numpy as np
import pandas as pd

import json

def main():

    # - import configs
    with open("configs.json", "r") as f:
        settings = json.load(f)

    # --- reproducibility / seeding ---
    seed = settings.get("seed", None)
    deterministic = settings.get("deterministic", False)
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    dataset = settings["dataset"]
    eval_split = settings["eval_split"]
    load_model = settings["load_model"]
    debias_lambda = settings["debias_lambda"]

    ssi_radius = settings["ssi_radius"]
    sri_radius = settings["sri_radius"]
    scale_grid = settings["sri_scale_grid"]
    distance_lag = settings["sri_distance_lag"]
    split_number = settings["sri_split_number"]

    trained_epochs = settings["trained_epochs"]
    debiased_epochs = settings["debiased_epochs"]
    epochs_to_train = settings["epochs_to_train"]
    epochs_to_debias = settings["epochs_to_debias"]

    loc_encoder_name = settings["loc_encoder_name"]
    loc_encoder_params = settings["loc_encoder_params"] # I want to use the information that should be used in here in the old checkpoint, such as the min frequency. They will be added after the checkpoint is loaded
    loc_encoder_params["spa_embed_dim"] = settings[dataset]["num_classes"]
    
    batch_size = settings["batch_size"]
    batch_count_print_avg_loss = settings["batch_count_print_avg_loss"]
    no_prior_hidden_dim = settings["no_prior_hidden_dim"]
    activation_func = settings["activation_func"]

    optimizer_lr = settings["optimizer_lr"]
    optimizer_weight_decay = settings["optimizer_weight_decay"]

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

    #if there is anything missing from the above entries, if anything extra is found from within the old_params, then the old shall replace the new, the ancient shall replace the temporary

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

    img_tr = torch.from_numpy(all_data["train_feats"]).float() # shape=(N, 2048)
    loc_tr = torch.from_numpy(all_data["train_locs"]).float() # shape=(N, 2) lon/lat in degrees
    y_tr = torch.from_numpy(all_data["train_classes"]).long() # shape=(N, )

    if loc_encoder_name == "rbf":
        loc_encoder_params["train_locs"] = all_data["train_locs"]
    
    img_te = torch.from_numpy(all_data["val_feats"]).float() # shape=(N, 2048)
    loc_te = torch.from_numpy(all_data["val_locs"]).float() # shape=(N, 2) lon/lat in degrees
    y_te = torch.from_numpy(all_data["val_classes"]).long() # shape=(N, )

    # --- sanity checks ---
    assert loc_tr.dtype == torch.float32 and loc_tr.ndim == 2 and loc_tr.shape[1] == 2, \
        f"loc_tr must be float32 (N,2); got {loc_tr.dtype} {loc_tr.shape}"
    assert loc_te.dtype == torch.float32 and loc_te.ndim == 2 and loc_te.shape[1] == 2, \
        f"loc_te must be float32 (N,2); got {loc_te.dtype} {loc_te.shape}"
    assert y_tr.dtype == torch.int64, f"y_tr must be long; got {y_tr.dtype}"
    assert y_te.dtype == torch.int64, f"y_te must be long; got {y_te.dtype}"

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
        loc_encoder = ThreeLayerMLP(input_dim = embed_dim, hidden_dim = no_prior_hidden_dim, category_count = num_classes, activation_func = activation_func).to(device)

    # - Criterion (select via registry; default to "embedding_loss" for backwards compatibility)
    train_loss_name = settings.get("train_loss_name", "embedding_loss")
    train_loss_params = settings.get("train_loss_params", {})
    criterion = get_loss(train_loss_name, train_loss_params if train_loss_params else None)

    # - Optimizer
    optimizer = Adam(params = loc_encoder.parameters(), lr = optimizer_lr, weight_decay = optimizer_weight_decay)
    ### torch.optim.Adam(self.loc_enc_model.parameters(),lr=params["lr"],weight_decay=params["weight_decay"],)
    print(len(optimizer.param_groups))
    for i, g in enumerate(optimizer.param_groups):
        print(i, len(g["params"]))
        print(g["params"])

    epochs_order = []

    if load_model:
        model_path = f"TorchSpatial/pre_trained_models/{loc_encoder_name.lower()}/model_{dataset}_{meta_type}_{loc_encoder_name}_trained{trained_epochs}_debiased{debiased_epochs}.pth.tar"

        ckpt = torch.load(model_path, map_location=device)
        state_dict = ckpt.get("state_dict", None)
        print(len(ckpt["optimizer"]["param_groups"]))
        for i, g in enumerate(ckpt["optimizer"]["param_groups"]):
            print(i, len(g["params"]))

        for i, (name, p) in enumerate(loc_encoder.named_parameters()):
            print(i, name, p.shape, p.requires_grad)
        
        # BEFORE: spa_enc.ffn.layers.0.layernorm.weight
        # AFTER: ffn.layers.0.layernorm.weight
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("spa_enc."):
                new_state_dict[k[len("spa_enc."):]] = v

        print(ckpt['optimizer']['param_groups'][0]['params'])
        print(ckpt["optimizer"]["state"].keys())

        loc_encoder.load_state_dict(new_state_dict, strict=True) # Old: "state_dict"
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
        except ValueError:
            print("optimizer ckpt failed to load (Original ckpt Cannot be Loaded Yet)")
        trained_epochs = ckpt.get("epoch", 0) - 1 # Old epoch is 31 when trained for 30
        if trained_epochs == -1: # Not old checkpoint
            ckpt["trained_epochs"] # New epoch can be loaded
        debiased_epochs = ckpt.get("debiased_epochs", 0) # Old: "debiased_epochs" not present, use 0; New: "debiased_epochs" present
        epochs_order = ckpt.get("epochs_order", [("train", trained_epochs)]) # Old was only trained regularly, never debiased
        old_params = ckpt.get("params", None)

        print(f"Checkpoint loaded from {model_path}; trained for {trained_epochs} epochs, debiased for {debiased_epochs} epochs, in the order of {epochs_order}")

        if old_params: # if there is something in there, then go and welcome it, invite it into the setup.
            print("======")
            print(old_params)
            print("======")
            print(f'Original TorchSpatial checkpoint contains parameters. If you need to adjust (ex. dataset/location encoder), change them manually on setup.')
            print("======")

    loc_encoder.train()

    ### Initialize gbs loss meta
    debias_loss = SSILoss()

    lats, lons = np.radians(loc_tr[:,1].numpy()), np.radians(loc_tr[:,0].numpy())
    train_partitioner = SSIPartitioner(np.array([lats, lons]).T, k=partition_k, radius=ssi_radius)
    train_perf_transformer = BinaryPerformanceTransformer(thres=BinaryPerformanceTransformer_thres)

    train(epochs=epochs_to_train,
        batch_count_print_avg_loss=batch_count_print_avg_loss,
        loc_encoder=loc_encoder,
        dataloader=train_loader,
        criterion=criterion,
        params = params,
        optimizer=optimizer,
        device=device)
    
    if epochs_to_train:
        trained_epochs += epochs_to_train
        epochs_order.append(("train", epochs_to_train))

    # - debias
    train_ssi_debias(epochs = epochs_to_debias,
        batch_count_print_avg_loss = batch_count_print_avg_loss,
        loc_encoder = loc_encoder,
        dataloader = train_loader,
        params = params,
        criterion = criterion,
        debias_loss = debias_loss,
        debias_lambda = debias_lambda,
        partitioner = train_partitioner,
        perf_transformer = train_perf_transformer,
        optimizer = optimizer,
        device = device)
        
    if epochs_to_debias:
        debiased_epochs += epochs_to_debias
        epochs_order.append(("debias", epochs_to_debias))

    # - save model
    model_path = f"TorchSpatial/pre_trained_models/{loc_encoder_name.lower()}/model_{dataset}_{meta_type}_{loc_encoder_name}_trained{trained_epochs}_debiased{debiased_epochs}.pth.tar"
    path = Path(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)


    torch.save({
        "trained_epochs": trained_epochs,
        "debiased_epochs": debiased_epochs,
        "epochs_order": epochs_order,
        "state_dict": loc_encoder.state_dict(),
        "optimizer": optimizer.state_dict(),
    }, path)

    print(f"Model saved as {model_path}; in total, trained for {trained_epochs} epochs, debiased for {debiased_epochs} epochs, in the order of {epochs_order}")
    
    # - test
    loc_encoder.eval()

    with torch.no_grad():

        lats, lons = np.radians(loc_te[:, 1].numpy()), np.radians(loc_te[:, 0].numpy())
        test_partitioner = SSIPartitioner(np.array([lats, lons]).T, k=partition_k, radius=ssi_radius)
        test_perf_transformer = BinaryPerformanceTransformer(thres=BinaryPerformanceTransformer_thres)

        rows = test(test_loader,
                    loc_encoder,
                    debias_loss,
                    test_partitioner,
                    test_perf_transformer,
                    device)

    df = pd.DataFrame(rows)
    eval_path = Path(f"TorchSpatial/eval_results/eval_{dataset}_{meta_type}_{eval_split}_{loc_encoder_name}_trained-{trained_epochs}_debiased-{debiased_epochs}.csv")
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(eval_path, index=True)

if __name__ == "__main__":
    main()