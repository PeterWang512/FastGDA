"""
Merge two feature sets into a combined feature set.

Usage:
    python merge.py --src1 dino --src2 clip_text --dest dino+clip_text --data-dir ./data/coco/feats
"""

import os
import argparse
import numpy as np


def merge_features(src1_dir, src2_dir, dest_dir):
    """Merge features from two source directories into destination directory."""
    
    # Create destination directory
    os.makedirs(dest_dir, exist_ok=True)
    
    # Merge data_feats.npy
    print(f"Merging data_feats.npy...")
    data_feats1 = np.load(os.path.join(src1_dir, 'data_feats.npy'))
    data_feats2 = np.load(os.path.join(src2_dir, 'data_feats.npy'))
    
    # Ensure they have the same number of samples
    assert data_feats1.shape[0] == data_feats2.shape[0], \
        f"Sample count mismatch: {data_feats1.shape[0]} vs {data_feats2.shape[0]}"
    
    data_feats = np.concatenate((data_feats1, data_feats2), axis=1)
    np.save(os.path.join(dest_dir, 'data_feats.npy'), data_feats)
    print(f"  {data_feats1.shape} + {data_feats2.shape} -> {data_feats.shape}")
    
    # Merge train_query_feats.npy
    print(f"Merging train_query_feats.npy...")
    train_query_feats1 = np.load(os.path.join(src1_dir, 'train_query_feats.npy'))
    train_query_feats2 = np.load(os.path.join(src2_dir, 'train_query_feats.npy'))
    
    assert train_query_feats1.shape[0] == train_query_feats2.shape[0], \
        f"Sample count mismatch: {train_query_feats1.shape[0]} vs {train_query_feats2.shape[0]}"
    
    train_query_feats = np.concatenate((train_query_feats1, train_query_feats2), axis=1)
    np.save(os.path.join(dest_dir, 'train_query_feats.npy'), train_query_feats)
    print(f"  {train_query_feats1.shape} + {train_query_feats2.shape} -> {train_query_feats.shape}")
    
    # Merge test_query_feats.npy
    print(f"Merging test_query_feats.npy...")
    test_query_feats1 = np.load(os.path.join(src1_dir, 'test_query_feats.npy'))
    test_query_feats2 = np.load(os.path.join(src2_dir, 'test_query_feats.npy'))
    
    assert test_query_feats1.shape[0] == test_query_feats2.shape[0], \
        f"Sample count mismatch: {test_query_feats1.shape[0]} vs {test_query_feats2.shape[0]}"
    
    test_query_feats = np.concatenate((test_query_feats1, test_query_feats2), axis=1)
    np.save(os.path.join(dest_dir, 'test_query_feats.npy'), test_query_feats)
    print(f"  {test_query_feats1.shape} + {test_query_feats2.shape} -> {test_query_feats.shape}")
    
    print(f"\n✓ Merged features saved to {dest_dir}/")


def main():
    parser = argparse.ArgumentParser(description='Merge two feature sets')
    parser.add_argument('--src1', type=str, required=True,
                        help='First feature type (e.g., dino)')
    parser.add_argument('--src2', type=str, required=True,
                        help='Second feature type (e.g., clip_text)')
    parser.add_argument('--dest', type=str, required=True,
                        help='Destination feature type (e.g., dino+clip_text)')
    parser.add_argument('--data-dir', type=str, default='./data/coco/feats',
                        help='Base feature directory (default: ./data/coco/feats)')
    
    args = parser.parse_args()
    
    # Build full paths
    src1_dir = os.path.join(args.data_dir, args.src1)
    src2_dir = os.path.join(args.data_dir, args.src2)
    dest_dir = os.path.join(args.data_dir, args.dest)
    
    # Validate source directories
    if not os.path.exists(src1_dir):
        raise ValueError(f"Source directory not found: {src1_dir}")
    if not os.path.exists(src2_dir):
        raise ValueError(f"Source directory not found: {src2_dir}")
    
    print(f"Merging features: {args.src1} + {args.src2} -> {args.dest}")
    print(f"Source 1: {src1_dir}")
    print(f"Source 2: {src2_dir}")
    print(f"Destination: {dest_dir}\n")
    
    merge_features(src1_dir, src2_dir, dest_dir)


if __name__ == '__main__':
    main()
