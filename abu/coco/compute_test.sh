#!/bin/bash

# Configuration - Edit these paths as needed
RESULT_DIR="results/test_influence"
FISHER_DIR="/mnt/localssd/ekfac_fisher"  # Path to Fisher matrix
MODEL_PATH="../../data/coco/model.bin"
SAMPLE_LATENT_PATH="../../data/coco/query_test/latents.npy"
SAMPLE_TEXT_PATH="../../data/coco/query_test/text_embeddings.npy"
SAMPLE_ROOT="../../data/coco/query_test/images"
PRETRAIN_LOSS_PATH="results/pretrain_loss.npy"
TODO_DIR="results/test_todo"
TOTAL_SAMPLES=110  # Update with actual number of test queries

# COCO dataset configuration
DATAROOT="../../data/coco"  # Root directory for COCO dataset

echo "Starting influence computation for test queries..."
echo "  Result directory: $RESULT_DIR"
echo "  TODO directory: $TODO_DIR"
echo "  Total samples: $TOTAL_SAMPLES"
echo ""

# Launch worker on each GPU
NUMGPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Launching workers on $NUMGPUS GPUs..."

for i in $(seq 0 $((NUMGPUS-1))); do
    CUDA_VISIBLE_DEVICES=$i python todo_worker.py \
        --result_dir "$RESULT_DIR" \
        --fisher_dir "$FISHER_DIR" \
        --model_path "$MODEL_PATH" \
        --sample_latent_path "$SAMPLE_LATENT_PATH" \
        --sample_text_path "$SAMPLE_TEXT_PATH" \
        --sample_root "$SAMPLE_ROOT" \
        --pretrain_loss_path "$PRETRAIN_LOSS_PATH" \
        --dataroot "$DATAROOT" \
        --todo_dir "$TODO_DIR" \
        --total_samples "$TOTAL_SAMPLES" &
done

# Wait for all workers to complete
wait $(jobs -p)
echo ""
echo "✓ All workers completed!"
echo ""
echo "Next step: Merge results using:"
echo "  python merge_test.py \\"
echo "    --root $RESULT_DIR \\"
echo "    --save_path ../data/coco/influence_test.pkl \\"
echo "    --num_samples $TOTAL_SAMPLES"

