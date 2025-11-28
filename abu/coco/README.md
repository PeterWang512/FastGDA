# COCO Influence Computation

This directory contains scripts to compute influence scores for COCO training and test datasets, generating `influence_train.pkl` and `influence_test.pkl` files used by FastGDA.

## Overview

The influence computation uses attribute-by-unlearning (ABU) to determine which training images are most influential for each query image. This is done by:
1. Unlearning each query image from a pre-trained diffusion model
2. Measuring the change in loss for each training image
3. Ranking training images by influence score

## Directory Structure

```
abu/coco/
├── README.md              # This file
├── compute_train.sh       # Generate influence_train.pkl
├── compute_test.sh        # Generate influence_test.pkl
├── merge_train.py         # Merge training influence results
├── merge_test.py          # Merge test influence results
└── todo_worker.py         # Worker script for parallel computation
```

## Prerequisites

Before running influence computation, download all required data files from [HuggingFace](https://huggingface.co/datasets/sywang/FastGDA) by running `../../scripts/download_data.sh`.

This will download:
1. **Pre-computed diffusion features** in `data/coco/query_train` and `data/coco/query_test`:
   - `latents.npy`: Latent codes for query images
   - `text_embeddings.npy`: Text embeddings for captions
   - `images/`: Directory of query images
   - `nn_dino.pkl`: Nearest neighbor data for training queries

2. **Pre-trained model and Fisher information**:
   - `data/coco/model.bin`: Model checkpoint
   - `data/coco/ekfac_fisher`: EKFAC Fisher matrix

### Additional Setup

You also need to generate latent codes and text embeddings from the training set:
```bash
python prepare_latents_text_embeds.py \
    --latents_path ../../data/coco/trainset/latents.npy \
    --text_embeds_path ../../data/coco/trainset/text_embeddings.npy
```

And collect diffusion loss of the training set from the pretrained model:
```bash
python compute_training_loss.py --output_path results/pretrain_loss.npy
```

## Usage

### Generate influence_train.pkl

```bash
# Edit compute_train.sh to set paths and parameters
# Then run:
bash compute_train.sh

# After all workers complete, merge results:
python merge_train.py \
    --root results/train_influence \
    --save_path ../../data/coco/influence_train.pkl \
    --num_samples <NUM_TRAIN_QUERIES>
```

### Generate influence_test.pkl

```bash
# Edit compute_test.sh to set paths and parameters
# Then run:
bash compute_test.sh

# After all workers complete, merge results:
python merge_test.py \
    --root results/test_influence \
    --save_path ../../data/coco/influence_test.pkl \
    --num_samples <NUM_TEST_QUERIES>
```

## Output Format

### influence_train.pkl

```python
{
    'random': {
        'influence_subset': ndarray (n_queries, n_subset),  # Influence scores for NN subset
        'rank_influence_subset': ndarray (n_queries, n_subset),  # Ranks within NN subset
        'rank': ndarray (n_queries, n_train)  # Full ranking of all training images
    }
}
```

### influence_test.pkl

```python
{
    'random': {
        'influence': ndarray (n_queries, n_train),  # Full influence scores
        'rank': ndarray (n_queries, n_train)  # Full ranking of all training images
    }
}
```

## Notes

- The computation is parallelized across multiple GPUs
- Each worker processes one query at a time using a todo/inprogress/complete task queue
- You might find `pkill` command useful if you want to kill the todo_workers
- Results are saved per-query and merged at the end

## Reference

This code is adapted from the [AttributeByUnlearning](https://github.com/PeterWang512/AttributeByUnlearning) repository for use with FastGDA.

