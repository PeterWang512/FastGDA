"""
Utility functions for FastGDA.
"""

import torch
import numpy as np
from tqdm import tqdm


@torch.no_grad()
def get_calibrated_feats(model, feats, batch_size=1000):
    """
    Apply model calibration (mlp) to features in batches.
    
    Args:
        model: DualMLPModel instance.
        feats: Input features [n_samples, n_features].
        batch_size: Batch size for processing.
        
    Returns:
        Calibrated features [n_samples, output_dim].
    """
    calibrated_feats = []
    for start in range(0, feats.shape[0], batch_size):
        end = min(start + batch_size, feats.shape[0])
        calibrated_feats.append(model.mlp(feats[start:end]))
    return torch.cat(calibrated_feats)


@torch.no_grad()
def get_rank(model, sample_feats, train_feats, batch_size=1000, per_query=False):
    """
    Compute influence scores and rankings.
    
    Args:
        model: Trained DualMLPModel.
        sample_feats: Query features [n_queries, n_features].
        train_feats: Training features [n_train, n_features] or [n_queries, n_train, n_features].
        batch_size: Batch size for calibration.
        per_query: If True, train_feats has shape [n_queries, n_train, n_features].
        
    Returns:
        influence: Influence scores.
        rank: Argsorted indices (low to high influence).
    """
    if per_query:
        # Each query has its own set of training samples
        infl_list = []
        rank_list = []
        calibrated_sample_feats = get_calibrated_feats(model, sample_feats, batch_size)
        
        for i in tqdm(range(calibrated_sample_feats.shape[0]), desc='Computing influence'):
            calibrated_train_feats = get_calibrated_feats(model, train_feats[i], batch_size)
            infl = model.get_score_from_reps(
                calibrated_sample_feats[[i]], 
                calibrated_train_feats
            ).cpu().numpy()[0]
            rank = np.argsort(infl)
            infl_list.append(infl)
            rank_list.append(rank)
        
        return infl_list, rank_list
    else:
        # All queries share the same training set
        calibrated_sample_feats = get_calibrated_feats(model, sample_feats, batch_size)
        calibrated_train_feats = get_calibrated_feats(model, train_feats, batch_size)
        infl = model.get_score_from_reps(calibrated_sample_feats, calibrated_train_feats).cpu().numpy()
        rank = np.argsort(infl, axis=1)
        return infl, rank


def setup_wandb(project_name="fastgda", entity=None, tags=None):
    """
    Setup Weights & Biases logging (optional).
    
    Args:
        project_name: W&B project name.
        entity: W&B entity/team name.
        tags: List of tags for the run.
        
    Returns:
        Boolean indicating if W&B was successfully initialized.
    """
    try:
        import wandb
        wandb.init(project=project_name, entity=entity, tags=tags or [])
        return True
    except ImportError:
        print("Warning: wandb not installed. Continuing without logging.")
        return False
    except Exception as e:
        print(f"Warning: Failed to initialize wandb: {e}")
        return False
