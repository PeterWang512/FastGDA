import os
import random
import numpy as np
import torch


class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataroot='data/coco', split='train', mode='no_flip', indices=None):
        self.dataroot = dataroot
        # New simplified path structure: data/coco/trainset/*.npy
        self.latents = np.load(os.path.join(dataroot, 'trainset', 'latents.npy'), mmap_mode='r')
        self.hidden_states = np.load(os.path.join(dataroot, 'trainset', 'text_embeddings.npy'), mmap_mode='r')
        self.num_captions = self.hidden_states.shape[1]

        self.mode = mode
        self.orig_length = len(self.hidden_states)
        if mode == 'no_flip':
            self.length = self.orig_length
            self.latents = self.latents[:self.length]
        elif mode == 'flip':
            self.length = self.orig_length
            self.latents = self.latents[self.length:]
        elif mode == 'no_flip_and_flip':
            self.length = self.orig_length * 2
        else:
            raise ValueError(f"Invalid mode: {mode}")

        if indices is not None:
            if self.mode == 'no_flip_and_flip':
                self.indices = indices + [i + self.orig_length for i in indices]
                self.length = len(self.indices)
            else:
                self.indices = indices
                self.length = len(indices)
        else:
            self.indices = None

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        actual_idx = self.indices[idx] if self.indices is not None else idx
        latents_tensor = torch.from_numpy(self.latents[actual_idx].copy()).float()

        hidx = actual_idx % len(self.hidden_states)
        hidden_states_tensor = torch.from_numpy(self.hidden_states[hidx].copy()).float()

        return latents_tensor, hidden_states_tensor
