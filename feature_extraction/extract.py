"""
Feature extraction script for FastGDA.

Extracts features from images or captions and saves them as .npy files.
Supports multiple feature extractors: dino, clip, clip_text.

For COCO trainset (with annotations), use extract_coco.py instead.
"""

import os
import sys
import argparse
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path

# Add parent directory to path to import models
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


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


def extract_image_features(image_dir, model, preprocess, model_type, device='cuda', batch_size=32):
    """Extract features from a directory of images.
    
    For COCO trainset with annotations, use extract_coco.py instead.
    """
    # Get all image files
    image_files = [f'{i}.png' for i in range(len(os.listdir(image_dir)))]
    
    if len(image_files) == 0:
        raise ValueError(f"No image files found in {image_dir}")
    
    print(f"Found {len(image_files)} images in {image_dir}")
    
    features = []
    
    # Process in batches
    for i in tqdm(range(0, len(image_files), batch_size), desc="Extracting image features"):
        batch_files = image_files[i:i+batch_size]
        batch_images = []
        
        for img_file in batch_files:
            img_path = os.path.join(image_dir, img_file)
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


def extract_text_features(caption_file, model, device='cuda', batch_size=32):
    """Extract features from a caption file (one caption per line).
    
    For COCO trainset with annotations, use extract_coco.py instead.
    
    Args:
        caption_file: Path to caption file (one caption per line)
        model: CLIP model
        device: Device to use
        batch_size: Batch size for processing
    
    Returns:
        numpy array of text features
    """
    import clip
    
    # Read captions from file
    if not os.path.isfile(caption_file):
        raise ValueError(f"Caption file not found: {caption_file}")
    
    with open(caption_file, 'r') as f:
        captions = [line.strip() for line in f if line.strip()]
    
    print(f"Found {len(captions)} captions in {caption_file}")
    
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


def main():
    parser = argparse.ArgumentParser(
        description='Extract features from images or text and save as .npy file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract DINO features from query images
  python extract.py --model dino --input data/coco/query_train/images --output features/dino/train_query_feats.npy
  
  # Extract CLIP text features from captions
  python extract.py --model clip_text --input data/coco/query_train/captions.txt --output features/clip_text/train_query_feats.npy

Note: For COCO trainset with annotations, use extract_coco.py instead.
"""
    )
    
    parser.add_argument('--model', type=str, required=True,
                        help='Model type: dino, clip, clip_text')
    parser.add_argument('--input', type=str, required=True,
                        help='Input: directory of images OR caption file (.txt)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output .npy file path')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for processing')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Load model
    model, preprocess = load_model(args.model, args.device)
    
    # Extract features
    if args.model == 'clip_text':
        features = extract_text_features(args.input, model, args.device, args.batch_size)
    else:
        features = extract_image_features(args.input, model, preprocess, args.model,
                                        args.device, args.batch_size)
    
    print(f"Features shape: {features.shape}")
    
    # Save
    np.save(args.output, features)
    print(f"✓ Saved to {args.output}")


if __name__ == '__main__':
    torch.set_grad_enabled(False)
    main()

