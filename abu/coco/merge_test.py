"""
Merge individual test influence results into influence_test.pkl.

This script collects per-query influence results and merges them into a single
pickle file with the format expected by FastGDA.
"""

import os
import argparse
import pickle
import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Merge test influence results into influence_test.pkl'
    )
    parser.add_argument('--root', type=str, required=True,
                        help='Directory containing influence_<i>.pkl files')
    parser.add_argument('--save_path', type=str, required=True,
                        help='Output path for merged influence_test.pkl')
    parser.add_argument('--num_samples', type=int, required=True,
                        help='Number of query samples')
    args = parser.parse_args()

    print(f"Merging {args.num_samples} influence results from {args.root}...")
    
    influence, rank = [], []
    
    for i in range(args.num_samples):
        pkl_path = os.path.join(args.root, f'influence_{i}.pkl')
        
        if not os.path.exists(pkl_path):
            raise FileNotFoundError(f"Missing result file: {pkl_path}")
        
        if (i + 1) % 100 == 0 or i == 0:
            print(f"  Loading {i+1}/{args.num_samples}...")
        
        with open(pkl_path, 'rb') as f:
            results = pickle.load(f)
        
        influence.append(results['influence'])
        rank.append(results['rank'])
    
    print("Stacking arrays...")
    influence = np.stack(influence, axis=0)
    rank = np.stack(rank, axis=0)
    
    print(f"  influence shape: {influence.shape}")
    print(f"  rank shape: {rank.shape}")
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    
    # Save merged results
    print(f"Saving to {args.save_path}...")
    with open(args.save_path, 'wb') as f:
        pickle.dump({
            'random': {
                'influence': influence,
                'rank': rank,
            }
        }, f)
    
    print("✓ Done!")

