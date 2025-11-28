#!/bin/bash

# Download all data files from HuggingFace for FastGDA
# Repository: https://huggingface.co/datasets/sywang/FastGDA
#
# Usage:
#   bash scripts/download_data.sh
#   

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
DATA_DIR="$PROJECT_ROOT/data"

echo "=========================================="
echo "Downloading FastGDA Data from HuggingFace"
echo "=========================================="
echo ""
echo "Repository: https://huggingface.co/datasets/sywang/FastGDA"
echo "Target directory: $DATA_DIR"
echo ""

# Check if huggingface-cli is installed
if ! command -v huggingface-cli &> /dev/null; then
    echo "Installing huggingface_hub..."
    pip install -q huggingface_hub[cli]
fi

# Create data directory
mkdir -p "$DATA_DIR"

# Download all files using huggingface-cli
echo "Downloading all files..."
echo ""
huggingface-cli download sywang/FastGDA \
    --repo-type dataset \
    --local-dir "$DATA_DIR" \

echo ""
echo "Download complete!"
echo ""

# Handle train2017.tar parts if they exist
if ls "$DATA_DIR"/coco/trainset/train2017.tar.part* 1> /dev/null 2>&1; then
    echo "=========================================="
    echo "Processing train2017.tar split files"
    echo "=========================================="
    echo ""
    
    # Concatenate parts
    echo "Concatenating train2017.tar parts..."
    cat "$DATA_DIR"/coco/trainset/train2017.tar.part* > "$DATA_DIR/coco/trainset/train2017.tar"
    echo "  → Created: $DATA_DIR/coco/train2017.tar"
    echo ""
    
    # Extract tar file
    echo "Extracting train2017.tar..."
    tar -xf "$DATA_DIR/coco/trainset/train2017.tar" -C "$DATA_DIR/coco/trainset/"
    echo "  → Extracted to: $DATA_DIR/coco/train2017/"
    echo ""
    
    # Clean up
    echo "Cleaning up temporary files..."
    rm "$DATA_DIR"/coco/trainset/train2017.tar.part*
    rm "$DATA_DIR/coco/trainset/train2017.tar"
    echo "  → Removed split files and tar archive"
    echo ""
fi

# move weights directory outside
mv "$DATA_DIR/coco/weights" "$PROJECT_ROOT/weights"


echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Data structure:"
echo "  $DATA_DIR/coco/"
echo "    ├── train2017/              # COCO training images"
echo "    ├── feats/                  # Precomputed features"
echo "    ├── influence_train.pkl     # Training influence rankings"
echo "    ├── influence_test.pkl      # Test influence rankings"
echo "    └── query_*/                # Query images and embeddings"
echo ""
echo "Next steps:"
echo "  1. Run demo: python demo.py --checkpoint weights/dino+clip_text.pth"
echo "  2. Train model: bash scripts/train_coco.sh"
echo "  3. Evaluate: bash scripts/eval_coco.sh"
echo ""
