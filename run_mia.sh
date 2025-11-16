#!/usr/bin/env bash
set -euo pipefail

CKPT="/home/user/weights/model.ckpt"
DATAROOT="/home/user/datasets/nuscenes/v1.0"                 # v1.0-{trainval} + samples/ sweeps/ maps/
MAPDIR="/home/user/datasets/nuscenes/map-labels-v1.3/datasets/nuscenes/map-labels-v1.3/"         # PNGs por token 

# sanity checks
[ -f "$CKPT" ] || { echo "ERRO: checkpoint não existe: $CKPT"; exit 1; }
[ -d "$DATAROOT" ] || { echo "ERRO: dataroot não existe: $DATAROOT"; exit 1; }
[ -d "$MAPDIR" ] || { echo "ERRO: map_dir não existe: $MAPDIR"; exit 1; }

# opcional: checar se há ao menos 1 PNG por token
#ls "$MAPDIR"/*.png >/dev/null 2>&1 || { echo "ERRO: nenhum *.png em $MAPDIR"; exit 1; }

WANDB_MODE=offline WANDB_SILENT=true HYDRA_FULL_ERROR=1 \
  python -m mapper.mapper -cn mapper_nuscenes \
    data.version=v1.0-trainval \
    data.data_dir="$DATAROOT" \
    data.map_dir="$MAPDIR" \
    training.checkpoint="$CKPT" \
    training.eval=true \
    +training.batch_size=1 \
    ++training.trainer.precision=16 \
    data.resize_image=384
set -euo pipefail
