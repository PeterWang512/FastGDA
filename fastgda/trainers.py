"""
Trainer classes for COCO and Stable Diffusion datasets.
"""

import os
import pickle
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from abc import ABC, abstractmethod

from .models import DualMLPModel
from .dataset import RankDatasetCOCO
from .utils import get_rank


class BaseTrainer(ABC):
    """
    Base trainer class with common functionality.
    """
    
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.model = None
        self.optimizer = None
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.rank_denom = self.args.nn_cutoff if self.args.random_subset_size is None else self.args.random_subset_size
        
    def create_model(self, n_features):
        """Create and initialize the model."""
        self.model = DualMLPModel(
            self.args.hidden_sizes,
            self.args.input_norm,
            self.args.dropout,
            n_features=n_features,
            n_outputs=self.args.out_feat_dim
        ).to(self.device)
        
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.lr)
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters())}")
        
    @abstractmethod
    def load_data(self):
        """Load dataset-specific features and ranks."""
        pass
    
    @abstractmethod
    def create_dataloaders(self):
        """Create training and validation dataloaders."""
        pass
    
    @abstractmethod
    def train_epoch(self, epoch):
        """Train for one epoch."""
        pass
    
    @abstractmethod
    def validate(self, epoch):
        """Run validation."""
        pass
    
    def train(self):
        """Main training loop."""
        print("\nStarting training...")
        for epoch in range(self.args.epochs):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate(epoch)
            
            print(f'Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')
            
            if self.args.wandb:
                import wandb
                wandb.log({'train_loss': train_loss, 'val_loss': val_loss, 'epoch': epoch})
    
    def save_model(self):
        """Save model checkpoint."""
        self.model.eval()
        self.model.to('cpu')
        model_path = os.path.join(self.args.output_dir, f'{self.args.name}.pth')
        torch.save(self.model.state_dict(), model_path)
        print(f"\nModel saved to {model_path}")
        return model_path


class COCOTrainer(BaseTrainer):
    """
    Trainer for MSCOCO dataset.
    """
    
    def load_data(self):
        """Load COCO features and ranks."""
        print(f"Loading COCO data...")
        
        # Load features - using new structure: data/coco/feats/{ftype}/
        feats_dir = os.path.join(self.args.data_dir, 'feats', self.args.ftype)
        data_path = os.path.join(feats_dir, 'data_feats.npy')
        query_path = os.path.join(feats_dir, 'train_query_feats.npy')  # New name!
        
        self.data_feats = torch.from_numpy(np.load(data_path)).to(self.device)
        self.query_feats = torch.from_numpy(np.load(query_path)).to(self.device)
        
        print(f"Data features shape: {self.data_feats.shape}")
        print(f"Query features shape: {self.query_feats.shape}")
        
        # Load ground truth ranks for training
        with open(self.args.rank_file, 'rb') as f:
            results = pickle.load(f)
            self.rank_gt = results['random']['rank']
        
        # Create model
        self.create_model(self.query_feats.shape[1])
        
    def create_dataloaders(self):
        """Create COCO dataloaders."""
        train_dataset = RankDatasetCOCO(
            self.rank_gt,
            cutoff=self.args.nn_cutoff,
            sample_start_end=(0, self.args.train_size),
            prob_outlier=self.args.prob_outlier,
            random_subset_size=self.args.random_subset_size,
            random_seed=self.args.random_seed
        )
        
        val_dataset = RankDatasetCOCO(
            self.rank_gt,
            cutoff=self.args.nn_cutoff,
            sample_start_end=(self.args.val_start, self.args.val_end),
            prob_outlier=0.0,
            random_subset_size=self.args.random_subset_size,
            random_seed=self.args.random_seed
        )
        
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True
        )
        
        self.val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.args.batch_size,
            shuffle=False
        )
        
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        
    def train_epoch(self, epoch):
        """Train one epoch on COCO."""
        self.model.train()
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch} | Training')
        
        total_loss = 0
        for i, j, r in pbar:
            sample_batch = self.query_feats[i]
            train_batch = self.data_feats[r]
            
            # Normalize rank positions to [0, 1] range as targets
            labels = (j.float() / self.rank_denom).to(self.device)
            
            self.optimizer.zero_grad()
            pred = self.model(sample_batch, train_batch)
            loss = self.loss_fn(pred, labels)
            
            pbar.set_description(f'Epoch {epoch} | Loss: {loss.item():.4f}')
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        pbar.close()
        return total_loss / len(self.train_loader)
        
    def validate(self, epoch):
        """Validate on COCO."""
        self.model.eval()
        val_losses = []
        pbar = tqdm(self.val_loader, desc=f'Epoch {epoch} | Validation')
        
        with torch.no_grad():
            for i, j, r in pbar:
                sample_batch = self.query_feats[i]
                train_batch = self.data_feats[r]
                
                labels = (j.float() / self.rank_denom).to(self.device)
                
                pred = self.model(sample_batch, train_batch)
                loss = self.loss_fn(pred, labels)
                val_losses.append(loss.item())
        
        pbar.close()
        return np.mean(val_losses)
