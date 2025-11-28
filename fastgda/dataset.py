"""
Dataset classes for FastGDA.
Separate classes for COCO and SD datasets.
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class RankDatasetCOCO(Dataset):
    """
    Dataset for training on ranked COCO data with optional outlier injection.
    
    Args:
        rank: Rank matrix [n_queries, n_candidates].
        cutoff: Maximum rank position to consider.
        sample_start_end: Tuple (start, end) to select query subset.
        prob_outlier: Probability of injecting outlier samples.
        dataset_size: Total size of the training dataset.
        random_subset_size: If set, randomly subsample this many candidates per query.
        random_seed: Random seed for reproducibility.
    """
    
    def __init__(self, rank, cutoff=10000, sample_start_end=None, prob_outlier=0.0, 
                 dataset_size=118287, random_subset_size=None, random_seed=1000):
        assert cutoff == rank.shape[1], "cutoff must be equal to the number of candidates in the rank matrix"
        if sample_start_end is None:
            self.start_idx, self.end_idx = 0, rank.shape[0]
            self.rank = rank
        else:
            self.start_idx, self.end_idx = sample_start_end
            self.rank = rank[self.start_idx:self.end_idx]
        
        self.cutoff = cutoff
        self.prob_outlier = prob_outlier

        # Prepare outlier list - COCO-specific
        if prob_outlier > 0:
            outlier_list = []
            full_indices = np.arange(dataset_size)
            for i in range(self.rank.shape[0]):
                outlier_list.append(np.setdiff1d(full_indices, self.rank[i]))
            self.outlier_list = np.array(outlier_list)

        # Create random subset if specified
        if random_subset_size is not None:
            generator = np.random.default_rng(random_seed)
            indices = [generator.choice(cutoff, random_subset_size, replace=False) 
                      for _ in range(self.rank.shape[0])]
            indices = np.sort(indices, axis=1)
            self.rank = self.rank[np.arange(self.rank.shape[0])[:, None], indices]
            self.cutoff = random_subset_size

    def __len__(self):
        assert len(self.rank.shape) == 2
        return self.rank.shape[0] * self.rank.shape[1]

    def __getitem__(self, idx):
        """
        Get a training sample.
        
        Returns:
            (query_idx, rank_pos, sample_idx)
        """
        i = idx // self.cutoff

        # Inject outliers with probability
        if self.prob_outlier > 0 and np.random.rand() < self.prob_outlier:
            j = self.cutoff - 1
            image_idx = np.random.choice(self.outlier_list[i], 1)[0]
            return i + self.start_idx, j, image_idx

        j = idx % self.cutoff
        return i + self.start_idx, j, self.rank[i, j]
