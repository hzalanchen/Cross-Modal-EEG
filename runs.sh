#!/bin/bash

# Train contrastive
seeds=(1024 2025 2048 3072 4096)

for seed in "${seeds[@]}"
do  
    echo "Running seed: $seed"
    python train_contrastive.py \
        --seed $seed \
        --exp_setting intra-subject \
        --lr 1e-4 \
        --train_epochs 50
done

# Train contrastive + Intra
for seed in "${seeds[@]}"
do
    echo "Running seed: $seed"
    python train_contrastive.py \
        --seed $seed \
        --exp_setting intra-subject \
        --lr 1e-4 \
        --train_epochs 50 \
        --geo-loss \
        --lambda1 5.0 \
        --geo-loss-dis cosine \
        --geo-last-epochs 25
done

# Train Decouple 
for seed in "${seeds[@]}"
do
    echo "Running seed: $seed"
    python train_decouple.py \
        --seed $seed \
        --exp_setting intra-subject \
        --lr 1e-4 \
        --train_epochs 50 \
        --lambda3 1.0 \
        --lambda4 1.0
done

# Train Decouple + Intra
for seed in "${seeds[@]}"
do
    echo "Running seed: $seed"
    python train_decouple.py \
        --seed $seed \
        --exp_setting intra-subject \
        --lr 1e-4 \
        --train_epochs 50 \
        --lambda3 1.0 \
        --lambda4 1.0 \
        --geo-loss \
        --lambda1 5.0 \
        --geo-loss-dis cosine \
        --geo-last-epochs 25
done


# Inter Subject 
# Train contrastive 
for seed in "${seeds[@]}"
do  
    echo "Running seed: $seed"
    python train_contrastive.py \
        --seed $seed \
        --exp_setting inter-subject \
        --lr 1e-4 \
        --train_epochs 10
done


# Inter Subject 
# Train contrastive + Intra
for seed in "${seeds[@]}"
do
    echo "Running seed: $seed"
    python train_contrastive.py \
        --seed $seed \
        --exp_setting inter-subject \
        --lr 1e-4 \
        --train_epochs 10 \
        --geo-loss \
        --lambda1 5.0 \
        --geo-loss-dis cosine \
        --geo-last-epochs 5
done