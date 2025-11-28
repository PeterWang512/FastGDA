"""
Unlearning loss function for influence computation.

This module provides the default unlearn loss used for computing influence scores
via attribute-by-unlearning.
"""

import torch
import torch.nn.functional as F


def get_loss_class(loss_type):
    """
    Get the loss class for the specified loss type.
    
    Args:
        loss_type (str): Type of loss function. Currently only 'unlearn' is supported.
    
    Returns:
        UnlearnLoss: The unlearn loss function instance.
    
    Raises:
        ValueError: If loss_type is not 'unlearn'.
    """
    if loss_type == 'unlearn':
        return UnlearnLoss()
    else:
        raise ValueError(f"Invalid loss type '{loss_type}'. Only 'unlearn' is supported.")


class UnlearnLoss(torch.nn.Module):
    """
    Standard unlearning loss function.
    
    This loss computes the negative MSE between the model's predicted noise
    and the actual noise added to the latents. The negative sign makes this
    a loss to maximize (i.e., minimize the match to the target), which is
    the goal of unlearning.
    """
    
    def forward(self, latents, conds, model, noise_scheduler):
        """
        Compute the unlearn loss.
        
        Args:
            latents (torch.Tensor): Clean latent representations
            conds (torch.Tensor): Conditioning (text embeddings)
            model: The diffusion model
            noise_scheduler: Scheduler for adding noise
        
        Returns:
            tuple: (loss_value, loss_name)
                - loss_value (torch.Tensor): Scalar loss value (negative MSE)
                - loss_name (str): Name of the loss ('unlearn_loss')
        """
        # Generate random noise
        noise = torch.randn_like(latents)
        bs = latents.shape[0]
        
        # Sample random timesteps
        timesteps = torch.randint(
            0, 
            noise_scheduler.num_train_timesteps, 
            (bs,)
        ).to(latents.device)
        
        # Add noise to latents
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
        
        # Get model prediction
        model_output = model(noisy_latents, timesteps, conds, return_dict=False)[0]
        
        # Return negative MSE (we want to maximize error = unlearn)
        return -F.mse_loss(model_output, noise), 'unlearn_loss'
