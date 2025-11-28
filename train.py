"""
Unified training script for FastGDA.
Supports both COCO and Stable Diffusion datasets.
"""

import os
import argparse
import torch

from fastgda.trainers import COCOTrainer
from fastgda.utils import setup_wandb


def parse_args():
    parser = argparse.ArgumentParser(description='Train FastGDA model for data attribution')
    
    # Dataset settings
    parser.add_argument('--mode', type=str, choices=['coco'], required=True,
                        help='Dataset mode')
    parser.add_argument('--ftype', type=str, default='dino+clip_text',
                        help='Feature type (e.g., dino+clip_text)')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Directory containing dataset files. Defaults to ./data/<mode>')
    parser.add_argument('--rank_file', type=str, default=None,
                        help='Path to ground truth rank file (.pkl). If not provided, uses data_dir/influence_train.pkl')
    
    # Training dataset settings
    parser.add_argument('--train_size', type=int, default=4900,
                        help='Number of training queries')
    parser.add_argument('--val_start', type=int, default=4900,
                        help='Validation set start index')
    parser.add_argument('--val_end', type=int, default=5000,
                        help='Validation set end index')
    parser.add_argument('--nn_cutoff', type=int, default=10000,
                        help='Maximum rank position to consider (number of nearest neighbors)')
    parser.add_argument('--prob_outlier', type=float, default=0.1,
                        help='Probability of injecting outlier samples')
    parser.add_argument('--random_subset_size', type=int, default=None,
                        help='If set, randomly subsample this many candidates per query')
    parser.add_argument('--random_seed', type=int, default=1000,
                        help='Random seed for reproducibility')
    
    # Model architecture
    parser.add_argument('--hidden_sizes', type=int, nargs='+', default=[768, 768, 768],
                        help='Hidden layer sizes')
    parser.add_argument('--input_norm', action='store_true',
                        help='Use layer normalization on input')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout probability')
    parser.add_argument('--out_feat_dim', type=int, default=768,
                        help='Output feature dimension')
    
    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4096,
                        help='Training batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    
    # Output settings
    parser.add_argument('--output_dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--name', type=str, required=True,
                        help='Experiment name')
    
    # Logging
    parser.add_argument('--wandb', action='store_true',
                        help='Enable Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='fastgda',
                        help='W&B project name')
    parser.add_argument('--wandb_entity', type=str, default=None,
                        help='W&B entity name')
    
    # Evaluation
    parser.add_argument('--eval_after_train', action='store_true',
                        help='Run evaluation after training')
    parser.add_argument('--topk_values', type=int, nargs='+', 
                        default=[500, 1000, 4000],
                        help='Top-k values for mAP (k) computation during evaluation')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set default data_dir if not provided
    if args.data_dir is None:
        args.data_dir = f'./data/{args.mode}'
        print(f"Using default data directory: {args.data_dir}")
    
    # Set default rank_file if not provided
    if args.rank_file is None:
        args.rank_file = os.path.join(args.data_dir, 'influence_train.pkl')
        print(f"Using default rank file: {args.rank_file}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup logging
    if args.wandb:
        setup_wandb(args.wandb_project, args.wandb_entity, [args.mode])
    
    # Initialize device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create appropriate trainer
    if args.mode == 'coco':
        trainer = COCOTrainer(args, device)
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")
    
    # Load data and create dataloaders
    trainer.load_data()
    trainer.create_dataloaders()
    
    # Train model
    trainer.train()
    
    # Save model
    trainer.save_model()
    
    # Run evaluation if requested
    if args.eval_after_train:
        from eval import run_evaluation
        
        os.makedirs(args.output_dir, exist_ok=True)
        eval_output = os.path.join(args.output_dir, f'{args.name}_eval.csv')
        run_evaluation(trainer.model, args, device, eval_output, args.topk_values)
    
    print("\nTraining complete!")


if __name__ == '__main__':
    main()
