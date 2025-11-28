"""
Model architectures for FastGDA.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FCModel(torch.nn.Module):
    """
    Fully connected neural network model with configurable layer sizes and activation.
    
    Args:
        sizes: List of hidden layer sizes (excluding input layer).
        input_norm: Whether to apply layer normalization on input.
        dropout: Dropout probability (default: 0.0).
        n_features: Number of input features.
        n_outputs: Number of output features.
    """
    
    def __init__(self, sizes, input_norm, dropout, n_features, n_outputs):
        super(FCModel, self).__init__()
        sizes.insert(0, n_features)
        layers = [nn.Linear(size_in, size_out) for size_in, size_out in zip(sizes[:-1], sizes[1:])]
        self.input_norm = nn.LayerNorm(n_features) if input_norm else nn.Identity()
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout or 0.0)
        self.output_size = sizes[-1]

        self.layers = nn.ModuleList(layers)
        self.output_layer = nn.Linear(sizes[-1], n_outputs)

        # Initialize weights with Xavier uniform
        for param in self.parameters():
            if len(param.shape) > 1:
                torch.nn.init.xavier_uniform_(param)

    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape [batch_size, n_features].
            
        Returns:
            Output tensor of shape [batch_size, n_outputs].
        """
        x = self.input_norm(x)
        for layer in self.layers:
            x = self.dropout(self.activation(layer(x)))
        x = self.output_layer(x)
        return x


class DualMLPModel(torch.nn.Module):
    """
    Simplified Dual MLP model for computing similarity scores between query and training samples.
    
    This model uses a single shared MLP to process both query and training samples,
    then computes cosine similarity between their representations with learnable scale and bias.
    
    Args:
        sizes: List of hidden layer sizes.
        input_norm: Whether to apply layer normalization.
        dropout: Dropout probability.
        n_features: Number of input features.
        n_outputs: Output embedding dimension.
    """
    
    def __init__(self, sizes, input_norm, dropout, n_features, n_outputs):
        super(DualMLPModel, self).__init__()
        # Single shared MLP for both query and training samples
        self.mlp = FCModel(sizes, input_norm, dropout, n_features, n_outputs)
        
        # Learnable scale and bias for cosine similarity
        self.scale = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))
    
    @property
    def mlp1(self):
        """Alias for backward compatibility."""
        return self.mlp
    
    @property
    def mlp2(self):
        """Alias for backward compatibility."""
        return self.mlp

    def forward(self, sample_batch, train_batch, output_rep=False):
        """
        Forward pass through the model.
        
        Args:
            sample_batch: Query sample features [batch_size, n_features].
            train_batch: Training sample features [batch_size, n_features].
            output_rep: Whether to return representations along with predictions.
            
        Returns:
            pred: Predicted similarity scores [batch_size].
            rep1, rep2: (Optional) Representations if output_rep=True.
        """
        rep1 = self.mlp(sample_batch)
        rep2 = self.mlp(train_batch)

        # Normalize representations
        rep1_normalized = F.normalize(rep1, p=2, dim=1)
        rep2_normalized = F.normalize(rep2, p=2, dim=1)
        
        # Compute cosine similarity with scale and bias
        cos_sim = torch.sum(rep1_normalized * rep2_normalized, dim=1)
        pred = cos_sim * self.scale + self.bias
        
        if output_rep:
            return pred, rep1, rep2
        return pred

    def get_score_from_reps(self, rep1, rep2):
        """
        Compute scores from pre-computed representations.
        
        Allows computing scores between different numbers of queries and training samples.
        
        Args:
            rep1: Query representations [n_queries, embedding_dim].
            rep2: Training representations [n_train, embedding_dim].
            
        Returns:
            Similarity scores [n_queries, n_train].
        """
        # Normalize representations
        rep1_normalized = F.normalize(rep1, p=2, dim=1)
        rep2_normalized = F.normalize(rep2, p=2, dim=1)
        
        # Compute cosine similarity matrix with scale and bias
        cos_sim = torch.mm(rep1_normalized, rep2_normalized.t())
        scores = cos_sim * self.scale + self.bias
        return scores

    def get_score(self, sample_batch, train_batch):
        """
        Compute scores directly from features.
        
        Args:
            sample_batch: Query features [batch_size, n_features].
            train_batch: Training features [batch_size, n_features].
            
        Returns:
            Similarity scores [batch_size].
        """
        return self.forward(sample_batch, train_batch)
