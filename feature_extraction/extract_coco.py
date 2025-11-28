"""
COCO-specific feature extraction functions.

This module handles feature extraction from COCO datasets using pycocotools.
Can be used as a library or run directly as a script.
"""

import os
import sys
import argparse
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

# Add parent directory to path to import models
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def is_coco_dataset(image_dir):
    """Check if directory is a COCO dataset with annotations."""
    # Check for COCO structure: train2017, val2017, etc.
    parent_dir = os.path.dirname(image_dir)
    annotations_dir = os.path.join(parent_dir, 'annotations')
    
    # Check if this looks like a COCO dataset
    if os.path.exists(annotations_dir) and os.path.basename(image_dir).endswith('2017'):
        return True
    
    # Also check if annotations is a sibling directory (for symlinks)
    if os.path.exists(os.path.join(image_dir, '..', 'annotations')):
        return True
    
    return False


def extract_image_features_coco(coco_dir, model, preprocess, model_type, device='cuda', batch_size=32):
    """Extract image features from COCO dataset using pycocotools."""
    try:
        from pycocotools.coco import COCO
    except ImportError:
        raise ImportError("pycocotools is required for COCO dataset. Install with: pip install pycocotools")
    
    # Determine split (train2017, val2017, etc.)
    split = os.path.basename(coco_dir)
    parent_dir = os.path.dirname(coco_dir)
    
    # Load COCO annotations
    ann_file = os.path.join(parent_dir, 'annotations', f'captions_{split}.json')
    if not os.path.exists(ann_file):
        raise ValueError(f"Annotation file not found: {ann_file}")
    
    print(f"Loading COCO annotations from {ann_file}...")
    coco = COCO(ann_file)
    img_ids = list(coco.imgs.keys())
    
    print(f"Found {len(img_ids)} images in COCO dataset")
    
    features = []
    
    # Process in batches
    for i in tqdm(range(0, len(img_ids), batch_size), desc="Extracting image features"):
        batch_ids = img_ids[i:i+batch_size]
        batch_images = []
        
        for img_id in batch_ids:
            img_dict = coco.loadImgs([img_id])[0]
            img_path = os.path.join(coco_dir, img_dict['file_name'])
            
            img = Image.open(img_path).convert('RGB')
            img_tensor = preprocess(img)
            batch_images.append(img_tensor)
        
        batch_tensor = torch.stack(batch_images).to(device)
        
        with torch.no_grad():
            if model_type == 'clip':
                feat = model.encode_image(batch_tensor).float()
            else:
                feat = model(batch_tensor)
            
            features.append(feat.cpu())
    
    features = torch.cat(features, dim=0)
    return features.numpy()


def extract_text_features_coco(coco_dir, model, device='cuda', batch_size=32):
    """Extract text features from COCO dataset using pycocotools.
    
    Uses the first caption for each image from COCO annotations.
    """
    try:
        from pycocotools.coco import COCO
        import clip
    except ImportError as e:
        raise ImportError(f"Required package not found: {e}. Install with: pip install pycocotools")
    
    # Determine split (train2017, val2017, etc.)
    split = os.path.basename(coco_dir)
    parent_dir = os.path.dirname(coco_dir)
    
    # Load COCO annotations
    ann_file = os.path.join(parent_dir, 'annotations', f'captions_{split}.json')
    if not os.path.exists(ann_file):
        raise ValueError(f"Annotation file not found: {ann_file}")
    
    print(f"Loading COCO annotations from {ann_file}...")
    coco = COCO(ann_file)
    img_ids = list(coco.imgs.keys())
    
    print(f"Found {len(img_ids)} images in COCO dataset")
    
    # Get captions for each image (use first caption)
    captions = []
    for img_id in img_ids:
        anns = coco.imgToAnns[img_id]
        if len(anns) > 0:
            captions.append(anns[0]['caption'])
        else:
            captions.append("")  # Empty caption if none found
    
    print(f"Loaded {len(captions)} captions")
    
    features = []
    
    # Process in batches
    for i in tqdm(range(0, len(captions), batch_size), desc="Extracting text features"):
        batch_captions = captions[i:i+batch_size]
        
        with torch.no_grad():
            tokens = clip.tokenize(batch_captions, truncate=True).to(device)
            feat = model.encode_text(tokens).float()
            features.append(feat.cpu())
    
    features = torch.cat(features, dim=0)
    return features.numpy()


def load_model(model_type, device='cuda'):
    """Load feature extraction model."""
    print(f"Loading {model_type} model...")
    
    if model_type == 'clip_text':
        import clip
        model, _ = clip.load("ViT-B/32", device=device)
        model.eval()
        preprocess = None
        return model, preprocess
    
    elif model_type == 'dino':
        from torchvision import transforms
        model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16', pretrained=True)
        model.eval().to(device)
        preprocess = transforms.Compose([
            transforms.Resize(256, interpolation=3),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        return model, preprocess
    
    elif model_type == 'clip':
        import clip
        model, preprocess = clip.load("ViT-B/32", device=device)
        model.eval()
        return model, preprocess
    
    else:
        raise ValueError(f"Unknown model type: {model_type}. Supported: dino, clip, clip_text")


def main():
    parser = argparse.ArgumentParser(
        description='Extract features from COCO dataset using pycocotools',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract DINO features from COCO trainset
  python extract_coco.py --model dino --input data/coco/trainset/train2017 --output features/dino/data_feats.npy
  
  # Extract CLIP text features from COCO trainset
  python extract_coco.py --model clip_text --input data/coco/trainset/train2017 --output features/clip_text/data_feats.npy
"""
    )
    
    parser.add_argument('--model', type=str, required=True,
                        help='Model type: dino, clip, clip_text')
    parser.add_argument('--input', type=str, required=True,
                        help='COCO dataset directory (e.g., train2017)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output .npy file path')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for processing (default: 32)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Validate input
    if not is_coco_dataset(args.input):
        raise ValueError(f"Input does not appear to be a COCO dataset: {args.input}")
    
    # Load model
    model, preprocess = load_model(args.model, args.device)
    
    # Extract features
    if args.model == 'clip_text':
        features = extract_text_features_coco(args.input, model, args.device, args.batch_size)
    else:
        features = extract_image_features_coco(args.input, model, preprocess, args.model, 
                                              args.device, args.batch_size)
    
    print(f"Features shape: {features.shape}")
    
    # Save
    np.save(args.output, features)
    print(f"✓ Saved to {args.output}")


if __name__ == '__main__':
    torch.set_grad_enabled(False)
    main()
