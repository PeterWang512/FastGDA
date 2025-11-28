import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from einops import rearrange
from opt_einsum import contract

from utils import load_safetensors, BufferDict


class NaturalGradientOptimizer(nn.Module):
    """Run natural gradient on a given model using diagonal or EKFAC Fisher information."""
    def __init__(self, fisher_type, fisher_dir, param_name_to_optimize, fisher_gain=None, damping_factor=0.1, damping_method='layerwise', project_rank=None):
        super(NaturalGradientOptimizer, self).__init__()

        self.param_name_to_optimize = param_name_to_optimize
        self.fisher_type = fisher_type

        # set damping
        self.damping_factor = damping_factor
        self.damping_method = damping_method

        # set project rank
        self.project_rank = project_rank

        # load fisher information
        if fisher_type == 'diagonal':
            self._load_diagonal_fisher(fisher_dir, fisher_gain, param_name_to_optimize)
        elif fisher_type == 'ekfac':
            self._load_ekfac_fisher(fisher_dir, fisher_gain, param_name_to_optimize)
        else:
            raise ValueError("Invalid Fisher type.")

    def optimization_step(self, model, step_size, grad_accum_steps):
        if self.fisher_type == 'diagonal':
            self._update_params_diagonal_fisher(model, step_size, grad_accum_steps)
        elif self.fisher_type == 'ekfac':
            self._update_params_ekfac_fisher(model, step_size, grad_accum_steps)
        else:
            raise ValueError("Invalid Fisher type.")

    def _update_params_diagonal_fisher(self, model, step_size, grad_accum_steps):
        assert self.project_rank is None, "Project rank not supported for diagonal Fisher yet."
        param_dict = dict(model.named_parameters())
        for k in self.fisher_entries:
            if f'module.{k}' in param_dict:  # handle cases like DDP model
                param = param_dict[f'module.{k}']
            else:
                param = param_dict[k]
            update_p = step_size * param.grad.double() / (self.fisher_info[k] * grad_accum_steps)
            param.add_(update_p)

    @torch.no_grad()
    def _update_params_ekfac_fisher(self, model, step_size, grad_accum_steps):
        param_dict = dict(model.named_parameters())
        grads = {k: v.grad for k, v in model.named_parameters() if any([entry in k for entry in self.fisher_entries])}
        for k in self.fisher_entries:
            # preprocess gradient to match the shape of EKFAC factors
            grad_weight, shape_key_record = self._get_processed_grad(grads, k, grad_accum_steps)

            # get EKFAC factors
            lamb_mat = self.lambda_matrix[k]
            act_eig = self.activation_eigenvectors[k]
            grad_eig = self.gradient_eigenvectors[k]

            # compute iHVP with EKFAC factors
            grad_weight = contract('im,ij,jn->mn', grad_eig, grad_weight, act_eig)
            if self.project_rank is not None:
                # find indices of lamb_mat that correspond to top ranks
                _, top_indices = torch.topk(lamb_mat.flatten(), self.project_rank)
                # print(f"Top indices: {top_indices}")
                # print(len(top_indices))

                # convert indices back to 2D and make mask
                # unfortunately, unravel_index is not in this version, code it manually
                top_indices = (top_indices // lamb_mat.shape[1], top_indices % lamb_mat.shape[1])
                # top_indices = torch.unravel_index(top_indices, lamb_mat.shape)
                top_rank_mask = torch.zeros_like(lamb_mat)
                top_rank_mask[top_indices[0], top_indices[1]] = 1

                # only keep top ranks of grad_weight and zero-out others
                grad_weight = grad_weight * top_rank_mask

            grad_weight.div_(lamb_mat)
            grad_weight = contract('mi,ij,nj->mn', grad_eig, grad_weight, act_eig)

            # unparse the gradient to match the shape of the model
            update_dict = self._unparse_grad(grad_weight, shape_key_record)
            for k, v in update_dict.items():
                param = param_dict[k]
                update_p = step_size * v
                param.add_(update_p)

    def _get_processed_grad(self, grads, module_key, grad_accum_steps):
        """Preprocess the gradient to match the shape of EKFAC factors."""
        # record the shape and key of the gradient
        shape_key_record = []

        # handle case for weight
        if f'module.{module_key}.weight' in grads:   # handle cases like DDP model
            module_key = f'module.{module_key}'
        assert f'{module_key}.weight' in grads, "Gradient for module not found."

        with torch.no_grad():
            grad_weight = grads[f'{module_key}.weight'] / grad_accum_steps
        shape_key_record.append((f'{module_key}.weight', grad_weight.shape))

        # handle case for convolution
        if len(grad_weight.shape) == 4:
            grad_weight = rearrange(grad_weight, 'o i ch cw -> o (i ch cw)')
        assert len(grad_weight.shape) == 2, "Gradient shape should be 2D."

        # handle case for bias
        if f'{module_key}.bias' in grads:
            with torch.no_grad():
                grad_bias = grads[f'{module_key}.bias'] / grad_accum_steps  # shape (out)
            grad_weight = torch.cat([grad_weight, grad_bias.unsqueeze(-1)], dim=-1)
            shape_key_record.append((f'{module_key}.bias', grad_bias.shape))

        return grad_weight, shape_key_record

    def _unparse_grad(self, update, shape_key_record):
        """Unparse the gradient to match the shape of the model."""
        assert len(shape_key_record) in [1, 2], f"Invalid shape_key_record length: {len(shape_key_record)}."

        # separate weight and bias if necessary
        if len(shape_key_record) == 2:
            grad_weight = update[:, :-1]
            grad_bias = update[:, -1]
        else:
            grad_weight = update

        update_dict = {}
        # unparse the weight
        weight_shape = shape_key_record[0][1]
        grad_weight = grad_weight.view(*weight_shape)
        update_dict[shape_key_record[0][0]] = grad_weight

        # unparse the bias
        if len(shape_key_record) == 2:
            update_dict[shape_key_record[1][0]] = grad_bias

        return update_dict

    def _load_diagonal_fisher(self, fisher_dir, fisher_gain, param_name_to_optimize):
        # Loading diagonal Fisher information.
        fisher_info = torch.load(f"{fisher_dir}/fisher_info.pt", map_location='cpu')
        self.fisher_entries = param_name_to_optimize
        trimmed_fisher_info = {}
        with torch.no_grad():
            for k, v in fisher_info.items():
                if k in self.fisher_entries:
                    fisher_layer = v * fisher_gain

                    # apply damping to fisher
                    if self.damping_method == 'layerwise':
                        damping = self.damping_factor * torch.mean(fisher_layer)
                    elif self.damping_method == 'global':
                        damping = self.damping_factor
                    elif self.damping_method is None or self.damping_factor == 0:
                        damping = 0
                    else:
                        raise ValueError("Invalid damping method.")

                    trimmed_fisher_info[k] = fisher_layer + damping

        self.fisher_info = BufferDict(trimmed_fisher_info)

    def _load_ekfac_fisher(self, fisher_dir, fisher_gain, param_name_to_optimize):
        # Loading Eigendecomposition results.
        activation_eigenvectors = load_safetensors(f"{fisher_dir}/activation_eigenvectors.safetensors")
        gradient_eigenvectors = load_safetensors(f"{fisher_dir}/gradient_eigenvectors.safetensors")

        # Loading Lambda matrices.
        lambda_matrix = load_safetensors(f"{fisher_dir}/lambda_matrix.safetensors")
        num_lambda_processed = load_safetensors(f"{fisher_dir}/num_lambda_processed.safetensors")

        # convert param_name_to_optimize to keys for EKFAC factors
        fisher_entries = list(set(['.'.join(name.split('.')[:-1]) for name in param_name_to_optimize]))
        assert all([entry in activation_eigenvectors for entry in fisher_entries]), "Invalid param_name_to_optimize."
        self.fisher_entries = fisher_entries

        # only store fisher entries
        self.activation_eigenvectors = BufferDict({k: v for k, v in activation_eigenvectors.items() if k in self.fisher_entries})
        self.gradient_eigenvectors = BufferDict({k: v for k, v in gradient_eigenvectors.items() if k in self.fisher_entries})
        trimmed_lambda_matrix = {}
        with torch.no_grad():
            for k, v in lambda_matrix.items():
                if k in self.fisher_entries:
                    lamb_layer = v * (fisher_gain / num_lambda_processed[k])

                    # apply damping to fisher
                    if self.damping_method == 'layerwise':
                        damping = self.damping_factor * torch.mean(lamb_layer)
                    elif self.damping_method == 'global':
                        damping = self.damping_factor
                    elif self.damping_method is None or self.damping_factor == 0:
                        damping = 0
                    else:
                        raise ValueError("Invalid damping method.")

                    trimmed_lambda_matrix[k] = lamb_layer + damping

        self.lambda_matrix = BufferDict(trimmed_lambda_matrix)
