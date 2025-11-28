#!/bin/bash
# Extract all COCO features: dino, clip, clip_text, and dino+clip_text

# Configuration (edit these as needed)
DATA_DIR="../data/coco"
FEAT_DIR="feats_test"  # Change to "feats" for production
BATCH_SIZE=256

# Feature types to extract
FTYPES=("dino" "clip" "clip_text")


echo "========================================================================"
echo "Extracting COCO features"
echo "========================================================================"
echo "Data directory: $DATA_DIR"
echo "Feature directory: $FEAT_DIR"
echo "Batch size: $BATCH_SIZE"
echo "Feature types: ${FTYPES[@]}"
echo "========================================================================"
echo ""

# Extract each feature type
for FTYPE in "${FTYPES[@]}"; do
    echo "========================================================================"
    echo "Extracting $FTYPE features"
    echo "========================================================================"
    
    OUTPUT_DIR="$DATA_DIR/$FEAT_DIR/$FTYPE"
    mkdir -p "$OUTPUT_DIR"
    
    # Extract training data features using extract_coco.py (COCO dataset)
    echo "Extracting training data features from COCO dataset..."
    python extract_coco.py \
        --model "$FTYPE" \
        --input "$DATA_DIR/trainset/train2017" \
        --output "$OUTPUT_DIR/data_feats.npy" \
        --batch-size $BATCH_SIZE
    
    # Extract query features using extract.py (regular image/caption files)
    if [ "$FTYPE" = "clip_text" ]; then
        # Text features - use caption files
        echo "Extracting training query features (captions)..."
        python extract.py \
            --model "$FTYPE" \
            --input "$DATA_DIR/query_train/captions.txt" \
            --output "$OUTPUT_DIR/train_query_feats.npy" \
            --batch-size $BATCH_SIZE
        
        echo "Extracting test query features (captions)..."
        python extract.py \
            --model "$FTYPE" \
            --input "$DATA_DIR/query_test/captions.txt" \
            --output "$OUTPUT_DIR/test_query_feats.npy" \
            --batch-size $BATCH_SIZE
    else
        # Image features
        echo "Extracting training query features (images)..."
        python extract.py \
            --model "$FTYPE" \
            --input "$DATA_DIR/query_train/images" \
            --output "$OUTPUT_DIR/train_query_feats.npy" \
            --batch-size $BATCH_SIZE
        
        echo "Extracting test query features (images)..."
        python extract.py \
            --model "$FTYPE" \
            --input "$DATA_DIR/query_test/images" \
            --output "$OUTPUT_DIR/test_query_feats.npy" \
            --batch-size $BATCH_SIZE
    fi
    
    echo ""
done

# Merge dino and clip_text to create dino+clip_text
echo "========================================================================"
echo "Merging dino and clip_text -> dino+clip_text"
echo "========================================================================"
python merge.py \
    --src1 dino \
    --src2 clip_text \
    --dest "dino+clip_text" \
    --data-dir "$DATA_DIR/$FEAT_DIR"

echo ""
echo "========================================================================"
echo "✓ All features extracted successfully!"
echo "========================================================================"
echo "Available features in $DATA_DIR/$FEAT_DIR/:"
echo "  - dino/"
echo "  - clip/"
echo "  - clip_text/"
echo "  - dino+clip_text/"
echo "========================================================================"
