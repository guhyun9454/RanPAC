#!/bin/bash

START_SEED=$1
END_SEED=$2
DATA_PATH=${3:-/local_datasets} 

for (( SEED=$START_SEED; SEED<=$END_SEED; SEED++ )); do
  python vil_main.py \
    --dataset iDigits \
    --num_tasks 20 \
    --data_path "$DATA_PATH" \
    --IL_mode vil \
    --seed $SEED \
    --ood_dataset EMNIST \
    --use_RP --M 10000 \
    --wandb_project iDigits_OODVIL_seed_tuning \
    --wandb_run "${SEED}_RanPAC"
done