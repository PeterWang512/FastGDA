#!/usr/bin/env python3
"""
Module for processing a single sample for unlearning and influence calculation.

This module refactors the original script into a function.
Call process_sample(args, sample_idx_override) to run unlearning for a given sample.
"""

import os
import pickle
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from loader import get_dataset
from models import get_model, print_param_info, get_optimizer
from unlearning.natural_gradient import NaturalGradientOptimizer
from unlearning.loss import get_loss_class
from compute_training_loss import collect_loss
from utils import COCOVis


def get_param_pattern_list(weight_selection):
    """
    Returns a list of parameter name patterns based on the weight_selection option.
    """
    if weight_selection == 'cross-attn-kv':
        param_pattern_list = [
            ['attn2'],
            ['to_k', 'to_v'],
        ]
    elif weight_selection == 'cross-attn':
        param_pattern_list = [
            ['attn2'],
        ]
    elif weight_selection == 'attn':
        param_pattern_list = [
            ['attn'],
        ]
    elif weight_selection == 'all':
        param_pattern_list = []
    else:
        raise ValueError(f'Invalid weight selection: {weight_selection}')
    return param_pattern_list


def get_subset_indices(nn_pkl, sample_idx, num_samples):
    with open(nn_pkl, 'rb') as f:
        subset_indices = pickle.load(f)['fixed']['rank'][sample_idx][:num_samples]
    return subset_indices.tolist()


def process_sample(args, sample_idx_override=None):
    """
    Process a single sample index for unlearning and influence calculation.
    
    If sample_idx_override is provided, it overrides the sample index in args.
    
    Args:
        args (argparse.Namespace): Contains all parameters (paths, hyperparameters, etc.).
        sample_idx_override (int, optional): A sample index to override args.sample_idx.
    """
    if sample_idx_override is not None:
        args.sample_idx = sample_idx_override

    # load model and noise scheduler
    model, noise_scheduler = get_model(args.task, model_path=args.model_path)
    model.to(args.device)

    # get parameters to update based on weight selection
    param_pattern_list = get_param_pattern_list(args.weight_selection)
    optimizer, param_names_to_optimize = get_optimizer(lr=0, model=model, optimizer_name='SGD',
                                                        param_pattern_list=param_pattern_list)
    print_param_info(model)

    # select subset if needed
    subset_indices = None
    if args.nn_pkl is not None:
        print('Loading subset indices...')
        subset_indices = get_subset_indices(args.nn_pkl, args.sample_idx, args.nn_num_samples)
        print(f'Subset indices loaded, length: {len(subset_indices)}')

    # load dataset and create data loader
    dataset = get_dataset(args.task, dataroot=args.dataroot, split='train', mode='no_flip_and_flip', indices=subset_indices)
    small_bs = args.loss_batch_size // dataset.num_captions // args.loss_time_samples
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=small_bs, shuffle=False)
    len_dataset = dataset.orig_length
    # print(len(dataset), len(data_loader))
    # exit()

    # initialize unlearn loss function
    unlearn_loss_fn = get_loss_class('unlearn')

    # initialize natural gradient optimizer
    natural_grad_optim = NaturalGradientOptimizer(
        fisher_type=args.fisher_type,
        fisher_dir=args.fisher_dir,
        param_name_to_optimize=param_names_to_optimize,
        fisher_gain=118287**2 if args.fisher_type == 'ekfac' else 118287,
        damping_factor=0.1,
        damping_method='layerwise',
        project_rank=None
    ).to(args.device)

    # load sample latents and text for unlearning
    latents = np.load(args.sample_latent_path)[[args.sample_idx]]
    latents = torch.from_numpy(latents).to(args.device).float().expand(args.unlearn_batch_size, -1, -1, -1)
    conds = np.load(args.sample_text_path)[[args.sample_idx], 0]
    conds = torch.from_numpy(conds).to(args.device).float().expand(args.unlearn_batch_size, -1, -1)

    # run unlearning steps
    print('Running unlearning...')
    for optim_step in range(args.unlearn_steps):
        optimizer.zero_grad()
        pbar = tqdm(range(args.unlearn_grad_accum_steps), desc=f'Unlearning step {optim_step + 1}/{args.unlearn_steps}')
        for step in pbar:
            loss, loss_name = unlearn_loss_fn(latents, conds, model, noise_scheduler)
            (-loss).backward()    # Add minus sign to match natural gradient convention

        with torch.no_grad():
            natural_grad_optim.optimization_step(model, args.unlearn_lr, args.unlearn_grad_accum_steps)

    optimizer.zero_grad()
    print('Unlearning done.')

    # collect loss for influence calculation
    print('Collecting loss...')
    with torch.no_grad():
        loss_no_flip_and_flip = collect_loss(
            model,
            data_loader,
            noise_scheduler,
            batch_size=args.loss_batch_size,
            time_samples=args.loss_time_samples,
            num_captions=dataset.num_captions,
            avg_timesteps=True,
            avg_captions=True,
            init_random_seed=0,
            device=args.device
        )
    print('Loss collected.')

    # influence calculation
    print('Calculating influence and visualize results...')
    if args.nn_pkl is None:
        influence = loss_no_flip_and_flip - np.load(args.pretrain_loss_path)
        assert influence.shape == (len_dataset * 2,)
        influence = np.maximum(influence[:len_dataset], influence[len_dataset:])

        results = {
            'influence': influence,
            'rank': np.argsort(-influence),
        }
    else:
        # pretrain_loss_indices = subset_indices + [i + len_dataset for i in subset_indices]
        # pretrain_loss = np.load(args.pretrain_loss_path)[pretrain_loss_indices]

        # have to recompute pretrain loss to make sure the random noise are the same
        # load model and noise scheduler
        model, noise_scheduler = get_model(args.task, model_path=args.model_path)
        model.to(args.device)

        # collect loss for influence calculation
        print('Collecting loss...')
        with torch.no_grad():
            pretrain_loss = collect_loss(
                model,
                data_loader,
                noise_scheduler,
                batch_size=args.loss_batch_size,
                time_samples=args.loss_time_samples,
                num_captions=dataset.num_captions,
                avg_timesteps=True,
                avg_captions=True,
                init_random_seed=0,
                device=args.device
            )
        print('Loss collected.')

        infl_subset = loss_no_flip_and_flip - pretrain_loss

        assert infl_subset.shape == (len(subset_indices) * 2,)
        infl_subset = np.maximum(infl_subset[:len(subset_indices)], infl_subset[len(subset_indices):])
        rank_influnece_subset = np.argsort(-infl_subset)
        rank = np.array(subset_indices)[rank_influnece_subset]
        # # initialize influence as an array of NaNs, of shape rank.shape[0], 118287
        # influence = np.full((rank.shape[0], 118287), -np.inf)
        # influence[:, subset_indices] = infl_subset

        results = {
            'rank': rank,
            'influence_subset': infl_subset,
            'rank_influence_subset': rank_influnece_subset,
        }

    # save results
    os.makedirs(args.result_dir, exist_ok=True)
    with open(os.path.join(args.result_dir, f'influence_{args.sample_idx}.pkl'), 'wb') as f:
        pickle.dump(results, f)
    print('Results saved.')

    # Optional: visualize top 10 influential images
    visualize_results(args, results)


def visualize_results(args, results):
    """Visualize top 10 influential images and save top 1000 captions."""
    try:
        # visualize top 10 influential images
        plt.figure(figsize=(20, 2))
        plt.subplot(1, 11, 1)
        plt.imshow(plt.imread(f'{args.sample_root}/{args.sample_idx}.png'))
        plt.title('Query')
        plt.axis('off')
        plt.subplots_adjust(wspace=0.01)

        coco_vis = COCOVis(path=args.dataroot, split='train')
        top_10_indices = results['rank'][:10]
        for i, idx in enumerate(top_10_indices):
            plt.subplot(1, 11, i + 2)
            plt.imshow(np.asarray(coco_vis[idx][0]))
            if 'rank_influence_subset' in results:
                rk_idx = results['rank_influence_subset'][i]
                plt.title(f'infl: {results["influence_subset"][rk_idx]:.2e}')
            else:
                plt.title(f'infl: {results["influence"][idx]:.2e}')
            plt.axis('off')

        plt.savefig(os.path.join(args.result_dir, f'visualization_{args.sample_idx}.jpg'))
        plt.close()

        # save captions of top 1000 influential images
        top_1000_indices = results['rank'][:1000]
        captions = []
        for idx in top_1000_indices:
            captions.append(coco_vis[idx][1][0].strip())
        with open(os.path.join(args.result_dir, f'captions_{args.sample_idx}.txt'), 'w') as f:
            f.write('\n'.join(captions))
    except Exception as e:
        print(f'Warning: Could not visualize results: {e}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Process a single sample for unlearning and influence calculation'
    )
    
    # Required arguments
    parser.add_argument('--result_dir', type=str, required=True, 
                        help='Directory to save results')
    parser.add_argument('--sample_latent_path', type=str, required=True, 
                        help='Path to sample latents (.npy)')
    parser.add_argument('--sample_text_path', type=str, required=True, 
                        help='Path to sample text embeddings (.npy)')
    parser.add_argument('--sample_idx', type=str, required=True, 
                        help='Sample index to process')
    parser.add_argument('--pretrain_loss_path', type=str, required=True, 
                        help='Path to pretrain loss (.npy)')
    
    # Optional: subset computation
    parser.add_argument('--nn_pkl', type=str, default=None, 
                        help='Path to nearest neighbors pickle (for subset computation)')
    parser.add_argument('--nn_num_samples', type=int, default=10000, 
                        help='Number of nearest neighbors to use for subset')
    
    # Model and data configuration
    parser.add_argument('--task', type=str, default='mscoco_t2i', 
                        help='Task name')
    parser.add_argument('--dataroot', type=str, default='data/coco', 
                        help='COCO dataset root directory (should contain trainset/ folder)')
    parser.add_argument('--sample_root', type=str, default='data/mscoco/sample', 
                        help='Sample images root for visualization')
    parser.add_argument('--model_path', type=str, default='data/mscoco/model.bin', 
                        help='Path to pretrained model checkpoint')
    parser.add_argument('--weight_selection', type=str, default='cross-attn-kv', 
                        help='Weight selection pattern (cross-attn-kv, cross-attn, attn, all)')
    
    # Fisher matrix configuration
    parser.add_argument('--fisher_type', type=str, default='ekfac', 
                        help='Type of Fisher matrix (ekfac)')
    parser.add_argument('--fisher_dir', type=str, default='data/ekfac_fisher', 
                        help='Directory containing Fisher matrix')
    
    # Unlearning hyperparameters
    parser.add_argument('--unlearn_lr', type=float, default=0.01, 
                        help='Learning rate for unlearning')
    parser.add_argument('--unlearn_steps', type=int, default=1, 
                        help='Number of unlearning optimization steps')
    parser.add_argument('--unlearn_batch_size', type=int, default=80, 
                        help='Batch size for unlearning')
    parser.add_argument('--unlearn_grad_accum_steps', type=int, default=625, 
                        help='Gradient accumulation steps for unlearning')
    
    # Loss computation hyperparameters
    parser.add_argument('--loss_batch_size', type=int, default=8000, 
                        help='Batch size for loss calculation')
    parser.add_argument('--loss_time_samples', type=int, default=20, 
                        help='Number of time samples to average over')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda', 
                        help='Device to use (cuda or cpu)')
    
    args = parser.parse_args()
    process_sample(args)
