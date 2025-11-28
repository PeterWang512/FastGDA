#!/bin/bash
# Example training script for MSCOCO dataset

python train.py \
    --mode coco \
    --ftype dino+clip_text \
    --train_size 4900 \
    --val_start 4900 \
    --val_end 5000 \
    --nn_cutoff 10000 \
    --prob_outlier 0.1 \
    --hidden_sizes 768 768 768 \
    --dropout 0.1 \
    --out_feat_dim 768 \
    --epochs 10 \
    --batch_size 4096 \
    --lr 0.001 \
    --name coco_baseline \
    --output_dir ./checkpoints \
    --eval_after_train \
    --topk_values 500 1000 4000

# Note: --data_dir defaults to ./data/coco
# Note: --rank_file defaults to ./data/coco/influence_train.pkl
# With --eval_after_train, will compute mAP (k) and save to:
#   - ./checkpoints/coco_baseline.pth (model)
#   - ./checkpoints/coco_baseline_eval.csv (mAP table)



