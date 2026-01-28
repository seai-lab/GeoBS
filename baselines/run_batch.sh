#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${BASE_DIR}/configs.json"
MAIN="${BASE_DIR}/main.py"
LOG_DIR="${BASE_DIR}/logs"

mkdir -p "${LOG_DIR}"

# Datasets to run
DATASETS=(
  # "inat_2017"
  "nabirds"
  "inat_2017"
)

# Models that can run without extra encoder params in configs.json
# (rbf/wrap/wrap_ffn/tile_ffn/Naive need extra params; omitted by request)
MODELS=(
  "Siren(SH)"
  "xyz"
  "rff"
  "Space2Vec-grid"
  "Space2Vec-theory"
  "NeRF"
  "Sphere2Vec-sphereC"
  "Sphere2Vec-sphereM"
  "Sphere2Vec-sphereC+"
  "Sphere2Vec-sphereM+"
  "Sphere2Vec-dfs"
)

BACKUP="${CONFIG}.bak"
cp -f "${CONFIG}" "${BACKUP}"
trap 'cp -f "${BACKUP}" "${CONFIG}"' EXIT

update_config() {
  local dataset="$1"
  local model="$2"
  python - "$CONFIG" "$dataset" "$model" <<'PY'
import json, sys
path, dataset, model = sys.argv[1], sys.argv[2], sys.argv[3]
with open(path, "r") as f:
    cfg = json.load(f)
cfg["dataset"] = dataset
cfg["loc_encoder_name"] = model
with open(path, "w") as f:
    json.dump(cfg, f, indent=4)
PY
}

for dataset in "${DATASETS[@]}"; do
  for model in "${MODELS[@]}"; do
    echo "=== dataset=${dataset} model=${model} ==="
    update_config "${dataset}" "${model}"
    log_file="${LOG_DIR}/${dataset}__${model//\//-}.log"
    python "${MAIN}" 2>&1 | tee "${log_file}"
  done
done
