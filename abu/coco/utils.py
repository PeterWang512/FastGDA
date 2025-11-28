"""
Utility functions for influence computation.

This module provides helper classes and functions used by the influence computation pipeline.
"""

import os
import re
import importlib
import torch
from PIL import Image
from safetensors import safe_open
from typing import Dict
from pycocotools.coco import COCO


class BufferDict(torch.nn.Module):
    """
    A dictionary-like object that registers all of its items as buffers.
    
    This is used for the Fisher matrix in natural gradient computation.
    A hack is used to prevent register_buffer error when seeing the '.' character in the key.
    """
    def __init__(self, input_dict):
        super().__init__()
        self._keys = list(input_dict.keys())
        for k, v in input_dict.items():
            self.register_buffer(self.encode_key(k), v)

    def encode_key(self, key):
        """Replace all '.' in a key with '&' to avoid register_buffer errors."""
        return key.replace(".", "&")

    def keys(self):
        """Return the original keys."""
        return self._keys

    def __getitem__(self, key):
        """Get item by original key."""
        return getattr(self, self.encode_key(key))


def load_safetensors(path: str) -> Dict[str, torch.Tensor]:
    """
    Load a dictionary of tensors from a safetensors file.
    
    Used for loading Fisher matrix data.

    Args:
        path (str): Path to the safetensors file.

    Returns:
        Dict[str, torch.Tensor]: Dictionary mapping parameter names to tensors.
    """
    tensor_dict = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensor_dict[key] = f.get_tensor(key)
    return tensor_dict


def find_and_import_module(folder_name, module_name):
    """
    Dynamically find and import a module from a folder.
    
    Used by the loader and models modules to dynamically load task-specific implementations.
    
    Args:
        folder_name (str): Folder path to search in (e.g., 'loader', 'models/diffusion')
        module_name (str): Name of the module to import (case-insensitive)
    
    Returns:
        module: The imported module
    
    Raises:
        ValueError: If the module is not found in the folder
    """
    found = False
    for module_path in os.listdir(folder_name):
        module_path = module_path.split(".")[0]
        if module_path.lower() == module_name.lower():
            found = True
            break

    if not found:
        raise ValueError(f"Cannot find module {module_name} in {folder_name}.")

    # Import the module dynamically
    parent_module = folder_name.replace("/", ".")
    full_module_name = f"{parent_module}.{module_path}"
    module = importlib.import_module(full_module_name)
    return module


def check_substrings(text, list_of_substrings):
    """
    Check if text contains all substrings from each sublist in list_of_substrings.
    
    Used for parameter name matching in model optimization.
    
    Args:
        text (str): Text to check
        list_of_substrings (list of list of str): List of substring lists
    
    Returns:
        bool: True if text matches all patterns, False otherwise
    
    Example:
        >>> check_substrings("model.attn2.to_k", [['attn2'], ['to_k', 'to_v']])
        True  # Contains 'attn2' AND ('to_k' OR 'to_v')
    """
    # Start building the regex pattern with lookaheads for each sublist
    pattern = ""
    for substrings in list_of_substrings:
        # Create a pattern segment for the current list of substrings
        substrings_pattern = f"({'|'.join(map(re.escape, substrings))})"
        # Add a lookahead for this pattern segment
        pattern += f"(?=.*{substrings_pattern})"

    # Use re.search to check if the text matches the full pattern
    if re.search(pattern, text):
        return True
    else:
        return False


class COCOVis:
    """
    COCO dataset visualization helper.
    
    Provides easy access to COCO images and captions for visualization purposes.
    """
    
    def __init__(self, path="data/coco", split="train"):
        """
        Initialize COCO visualization helper.
        
        Args:
            path (str): Path to COCO dataset root directory
            split (str): Dataset split ('train' or 'val')
        """
        # Path structure: data/coco/trainset/train2017/ and data/coco/annotations/
        dataType = f"{split}2017"
        annFile = os.path.join(path, "trainset", "annotations", f"captions_{dataType}.json")
        self.imgdir = os.path.join(path, "trainset", dataType)
        self.coco = COCO(annFile)
        self.img_ids = list(self.coco.imgs.keys())
        self.captions = self.coco.imgToAnns

    def __getitem__(self, idx):
        """
        Get image and captions by index.
        
        Args:
            idx (int): Index into the dataset
        
        Returns:
            tuple: (image, captions)
                - image (PIL.Image): Center-cropped RGB image
                - captions (list of str): List of captions for this image
        """
        # Get image and caption
        i = self.img_ids[idx]
        img_path = os.path.join(self.imgdir, self.coco.loadImgs(i)[0]['file_name'])
        img = Image.open(img_path)
        img = img.convert('RGB')

        # Center crop to square
        w, h = img.size
        if w > h:
            img = img.crop(((w - h) // 2, 0, (w + h) // 2, h))
        elif h > w:
            img = img.crop((0, (h - w) // 2, w, (h + w) // 2))

        captions = [x["caption"] for x in self.captions[i]]
        return img, captions
