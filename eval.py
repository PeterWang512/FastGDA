"""
Unified evaluation script for FastGDA.
Computes influence scores and rankings and mAP@k on test sets.
"""

import os
import pickle
import argparse
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import average_precision_score

from fastgda import DualMLPModel
from fastgda.utils import get_rank


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate FastGDA model')
    
    # Model and checkpoint
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (.pth)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output path for results (.csv)')
    
    # Dataset mode
    parser.add_argument('--mode', type=str, choices=['coco'], required=True,
                        help='Dataset mode')
    parser.add_argument('--ftype', type=str, required=True,
                        help='Feature type')
    
    # Data paths
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Directory containing dataset files. Defaults to ./data/<mode>')
    
    # Model architecture (must match training)
    parser.add_argument('--hidden_sizes', type=int, nargs='+', default=[768, 768, 768],
                        help='Hidden layer sizes')
    parser.add_argument('--input_norm', action='store_true',
                        help='Use layer normalization on input')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout probability')
    parser.add_argument('--out_feat_dim', type=int, default=768,
                        help='Output feature dimension')
    
    # Evaluation settings  
    parser.add_argument('--topk_values', type=int, nargs='+', 
                        default=[500, 1000, 4000],
                        help='Top-k values for mAP computation')
    
    return parser.parse_args()


def load_model(args, device):
    """Load trained model from checkpoint."""
    # Infer n_features from checkpoint
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    
    # Assume new format checkpoint
    if 'mlp.layers.0.weight' not in checkpoint:
        raise ValueError("Checkpoint appears to be in old format. Please convert using convert_checkpoints.py")
    
    n_features = checkpoint['mlp.layers.0.weight'].shape[1]
    
    # Create model
    model = DualMLPModel(
        args.hidden_sizes,
        args.input_norm,
        args.dropout,
        n_features=n_features,
        n_outputs=args.out_feat_dim
    ).to(device)
    
    # Load state dict
    model.load_state_dict(checkpoint, strict=True)
    model.eval()
    
    print(f"✓ Loaded model with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    return model


def load_coco_data(args, device):
    """Load COCO dataset features."""
    # New structure: data/coco/feats/{ftype}/
    feats_dir = os.path.join(args.data_dir, 'feats', args.ftype)
    
    data_path = os.path.join(feats_dir, 'data_feats.npy')
    test_query_path = os.path.join(feats_dir, 'test_query_feats.npy')
    
    data_feats = torch.from_numpy(np.load(data_path)).to(device)
    test_query_feats = torch.from_numpy(np.load(test_query_path)).to(device)
    
    return data_feats, test_query_feats


def compute_map_k(rank_pred, rank_gt, topk):
    """Compute mAP (k) given predicted and ground truth rankings.
    
    Args:
        rank_pred: Predicted rankings [num_queries, num_candidates]
        rank_gt: Ground truth rankings [num_queries, num_candidates]
        topk: Top-k value for mAP computation
        
    Returns:
        mean_ap: Mean average precision
        std_err: Standard error
    """
    num_queries = rank_pred.shape[0]
    scores = []
    
    for i in range(num_queries):
        # Get top-k for just the ground truth reference ranking
        rk_ref = rank_gt[i, :topk]
        rk_pred = rank_pred[i]
        
        # Binary labels: 1 if predicted item is in ground truth, 0 otherwise
        y_true = np.isin(rk_pred, rk_ref).astype(int)
        
        # Scores: 1/(rank+1) for each position
        y_score = 1.0 / (np.arange(len(rk_pred)) + 1)
        
        # Compute AP
        scores.append(average_precision_score(y_true, y_score))
    
    mean_ap = np.mean(scores)
    std_err = np.std(scores) / np.sqrt(num_queries)
    
    return mean_ap, std_err


def evaluate_coco(model, data_feats, test_query_feats, rank_gt, topk_values):
    """Run evaluation for COCO dataset."""
    print("\nEvaluating on test set...")
    
    # Get predicted rankings
    infl, rank_pred = get_rank(model, test_query_feats, data_feats, per_query=False)
    
    # Compute mAP (k) for different k values
    map_results = {}
    for topk in topk_values:
        mean_ap, std_err = compute_map_k(rank_pred, rank_gt, topk)
        map_results[topk] = {'mean': mean_ap, 'std_err': std_err}
        print(f"  mAP ({topk}): {mean_ap:.4f} ± {std_err:.4f}")
    
    results = {
        'test': {
            'influence': infl,
            'rank': rank_pred,
            'map': map_results
        }
    }
    
    return results


def save_map_table(results, output_path):
    """Save mAP results as a CSV table."""
    import csv
    
    csv_path = output_path.replace('.pkl', '_map.csv')
    
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['k', 'mAP (k)', 'std_err'])
        
        map_results = results['test']['map']
        for k in sorted(map_results.keys()):
            mean_ap = map_results[k]['mean']
            std_err = map_results[k]['std_err']
            writer.writerow([k, f'{mean_ap:.6f}', f'{std_err:.6f}'])
    
    print(f"mAP table saved to {csv_path}")


def run_evaluation(model, args, device, output_path, topk_values=[500, 1000, 2000, 4000]):
    """
    Run complete evaluation pipeline and save results as CSV.
    
    This is a high-level function that can be called from train.py or standalone scripts.
    
    Args:
        model: Trained DualMLPModel
        args: Namespace with mode, data_dir, ftype attributes
        device: torch device
        output_path: Path to save results (.csv)
        topk_values: List of k values for mAP (k) computation
    
    Returns:
        results: Dictionary with evaluation results
    """
    print("\n" + "="*70)
    print("Running evaluation...")
    print("="*70)
    
    # Set default data_dir if not provided
    if not hasattr(args, 'data_dir') or args.data_dir is None:
        args.data_dir = f'./data/{args.mode}'
        print(f"Using default data directory: {args.data_dir}")
    
    # Load test ground truth rankings
    test_rank_file = os.path.join(args.data_dir, 'influence_test.pkl')
    print(f"Loading ground truth rankings from {test_rank_file}...")
    with open(test_rank_file, 'rb') as f:
        rank_data = pickle.load(f)
        rank_gt = rank_data['random']['rank']
    print(f"Ground truth rank shape: {rank_gt.shape}")
    
    # Set model to eval mode
    model.eval()
    model.to(device)
    
    # Load data and run evaluation based on mode
    if args.mode == 'coco':
        print("\nLoading COCO test data...")
        data_feats, test_query_feats = load_coco_data(args, device)
        print(f"Data shape: {data_feats.shape}")
        print(f"Test query shape: {test_query_feats.shape}")
        
        results = evaluate_coco(model, data_feats, test_query_feats, 
                               rank_gt, topk_values)
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")
    
    # Save mAP results as CSV
    save_map_table(results, output_path)
    
    # Print summary
    print(f"\nmAP (k) results:")
    for k in sorted(results['test']['map'].keys()):
        mean_ap = results['test']['map'][k]['mean']
        std_err = results['test']['map'][k]['std_err']
        print(f"  mAP ({k}): {mean_ap:.4f} ± {std_err:.4f}")
    
    return results


def main():
    args = parse_args()
    
    # Set default data_dir if not provided
    if args.data_dir is None:
        args.data_dir = f'./data/{args.mode}'
        print(f"Using default data directory: {args.data_dir}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = load_model(args, device)
    
    # Run evaluation
    run_evaluation(model, args, device, args.output, args.topk_values)


if __name__ == '__main__':
    main()
