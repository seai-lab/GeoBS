# GeoBS TorchSpatial Baseline

Rewriting the interface of the TorchSpatial library, including a new simplified Encoder Selector, a simplified training loop, simplified model objects, and a clear organization.

---

## How to run

All entry scripts must be executed from the **`baselines/`** directory (the directory that *contains* `TorchSpatial/`, not `TorchSpatial/` itself):

```bash
cd baselines

# Standard training + optional SSI debiasing
python main.py

# Training + SSI (Spatial Self-Information) debiasing
python main_ssi.py

# Training + SRI (Spatial Relative-Information) debiasing
python main_sri.py
```

All configuration lives in `baselines/configs.json`.

---

## Key `configs.json` options

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `"dataset"` | string | `"birdsnap"` | Dataset to use (birdsnap, nabirds, inat_2017, inat_2018, fmow, yfcc) |
| `"loc_encoder_name"` | string | `"Space2Vec-grid"` | Location encoder to use (see Encoder section) |
| `"loc_encoder_params"` | object | — | Encoder hyper-parameters passed directly to `get_loc_encoder()` |
| `"epochs_to_train"` | int | `0` | How many regular training epochs to run |
| `"epochs_to_debias"` | int | `0` | How many debiasing epochs to run |
| `"load_model"` | bool | `true` | Whether to load a checkpoint before training |
| `"train_loss_name"` | string | `"embedding_loss"` | Name of the training loss (see Loss section) |
| `"train_loss_params"` | object | `{}` | Extra kwargs forwarded to the loss function |
| `"seed"` | int \| null | `42` | Random seed for `random`, `numpy`, `torch`, `torch.cuda` |
| `"deterministic"` | bool | `false` | Set `torch.backends.cudnn.deterministic = True` and `benchmark = False` |
| `"optimizer_lr"` | float | `0.001` | Adam learning rate |

---

## Selecting a different loss

The training loss is resolved through a small registry in
`TorchSpatial/utils/loss_registry.py`.  The default is `"embedding_loss"`,
the contrastive embedding objective from the TorchSpatial paper.

### Using a built-in loss

Set `"train_loss_name"` in `configs.json`:

```json
{
    "train_loss_name": "embedding_loss",
    "train_loss_params": {}
}
```

### Registering a custom loss

1. Write a function with the signature:

   ```python
   def my_loss(model, params, loc_feat, loc_class, user_ids, inds, **kwargs):
       ...
       return loss  # scalar tensor
   ```

2. Register it at the top of your entry script (before `main()` is called):

   ```python
   from TorchSpatial.utils.loss_registry import register_loss
   from my_losses import my_loss

   register_loss("my_loss", my_loss)
   ```

3. Set `"train_loss_name": "my_loss"` (and any kwargs under `"train_loss_params"`)
   in `configs.json`.

---

## Gotchas for matching upstream TorchSpatial results

1. **Float coordinates** — The original TorchSpatial pipeline keeps lon/lat as
   `float32`.  This baseline previously cast `loc_tr`/`loc_te` to `torch.long`,
   destroying decimal precision.  That bug is now fixed.

2. **Lon/lat order** — Both this baseline and the original TorchSpatial use
   `[lon, lat]` order (column 0 = longitude, column 1 = latitude).
   `np.radians(loc[:, 1])` gives latitudes in radians (used for GBS
   partitioners), `np.radians(loc[:, 0])` gives longitudes.

3. **Checkpoint compatibility** — Original TorchSpatial checkpoints store
   weights under `"spa_enc.*"` keys.  `main.py` strips this prefix
   automatically before calling `load_state_dict`.  `main_ssi.py` /
   `main_sri.py` try `"loc_encoder"` first, then fall back to `"state_dict"`.

4. **Device** — `device` is computed automatically as
   `torch.device("cuda" if torch.cuda.is_available() else "cpu")`.
   The `"device"` key inside `loc_encoder_params` in `configs.json` is
   *overwritten* at runtime; any string value there is only a reminder, not
   the final device.

5. **Seeds / determinism** — Set `"seed"` (integer) and `"deterministic": true`
   in `configs.json` to get fully reproducible runs.  Note that
   `torch.backends.cudnn.deterministic = True` can slow down training.

6. **eval_results directory** — The CSV is written to
   `TorchSpatial/eval_results/`.  The directory is created automatically if it
   does not exist.

