#!/bin/bash

START_SEED=$1
END_SEED=$2
DATA_PATH=${3:-/local_datasets} 

for (( SEED=$START_SEED; SEED<=$END_SEED; SEED++ )); do
  python vil_main.py \
    --dataset DomainNet \
    --num_tasks 30 \
    --data_path "$DATA_PATH" \
    --IL_mode vil \
    --seed $SEED \
    --ood_dataset CORe50 \
    --use_RP --M 10000 \
    --wandb_project DomainNet_OODVIL_seed_tuning \
    --wandb_run "${SEED}_RanPAC"
done