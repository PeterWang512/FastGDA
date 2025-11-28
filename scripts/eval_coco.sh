#!/bin/bash
# Example evaluation script for MSCOCO

python eval.py \
    --mode coco \
    --ftype dino+clip_text \
    --checkpoint ./weights/dino+clip_text.pth \
    --output ./results/dino+clip_text.csv \
    --topk_values 500 1000 4000

# Note: --data_dir defaults to ./data/coco
# Ground truth rankings loaded from ./data/coco/influence_test.pkl
# This will compute mAP (k) and save results to:
#   - ./results/coco_baseline.csv (mAP table)



