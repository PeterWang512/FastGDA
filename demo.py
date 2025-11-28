"""
Gradio demo for FastGDA image attribution on MSCOCO dataset.
"""

import argparse
import os
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import clip
import gradio as gr

from fastgda import DualMLPModel
from feature_extraction.extract import load_model
from abu.coco.utils import COCOVis


def parse_args():
    parser = argparse.ArgumentParser(description='FastGDA Demo')
    parser.add_argument('--checkpoint', type=str, default='weights/dino+clip_text.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, default='data/coco',
                        help='Path to MSCOCO data directory')
    parser.add_argument('--feature_dir', type=str, default='data/coco/feats/dino+clip_text',
                        help='Path to precomputed features')
    parser.add_argument('--port', type=int, default=7860,
                        help='Port to run the demo on')
    return parser.parse_args()


class AttributionDemo:
    """Demo class for image attribution."""
    
    def __init__(self, checkpoint_path, data_dir, feature_dir):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        # Load image feature extractor
        print(f"Loading DINO model...")
        self.image_model, self.image_preprocess = load_model('dino', device=self.device)
        
        # Load CLIP for text features
        print("Loading CLIP Text model...")
        self.text_model, _ = load_model('clip_text', device=self.device)
        
        # Load trained attribution model
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        n_features = checkpoint['mlp.layers.0.weight'].shape[1]
        
        self.model = DualMLPModel(
            [768, 768, 768],
            input_norm=False,
            dropout=0.1,
            n_features=n_features,
            n_outputs=768
        ).to(self.device)
        
        # Load state dict
        self.model.load_state_dict(checkpoint, strict=True)
        self.model.eval()
        print("✓ Model loaded successfully")
        
        # Load precomputed MSCOCO training features
        print("Loading MSCOCO training features...")
        data_feats_path = os.path.join(feature_dir, 'data_feats.npy')
        src_data_feats = np.load(data_feats_path)
        src_data_feats = torch.from_numpy(src_data_feats).to(self.device)
        
        # Precompute calibrated features
        print("Calibrating features...")
        calib_data_feats = []
        with torch.no_grad():
            for start in range(0, src_data_feats.shape[0], 1000):
                end = min(start + 1000, src_data_feats.shape[0])
                out = self.model.mlp(src_data_feats[start:end])
                calib_data_feats.append(out)
        self.calib_data_feats = torch.cat(calib_data_feats)
        self.calib_data_feats = F.normalize(self.calib_data_feats, dim=-1)
        
        # Setup COCO visualizer
        print("Setting up COCO data...")
        self.coco_vis = COCOVis(path=data_dir, split='train')

        # Load sample captions
        caption_path = os.path.join(data_dir, 'query_test', 'captions.txt')
        with open(caption_path, 'r') as f:
            self.sample_captions = [line.strip() for line in f]
        
        self.data_dir = data_dir
        print("Initialization complete!")
    
    def predict(self, idx, top_k=10):
        """
        Predict top-k attributed images for a given query index.
        
        Args:
            idx: Query sample index.
            top_k: Number of top attributed images to return.
            
        Returns:
            Query image, caption, and list of attributed images.
        """
        idx = int(idx)
        
        # Load query image and caption
        sample_path = os.path.join(self.data_dir, 'query_test', 'images', f'{idx}.png')
        if not os.path.exists(sample_path):
            return None, "Sample not found", []
        
        query_img = Image.open(sample_path).convert('RGB')
        sample_caption = self.sample_captions[idx] if idx < len(self.sample_captions) else ""
        
        # Compute image and text features
        img_input = self.image_preprocess(query_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            img_feat = self.image_model(img_input).float()

        text_tokens = clip.tokenize([sample_caption]).to(self.device)
        with torch.no_grad():
            text_feat = self.text_model.encode_text(text_tokens).float()
        
        # Concatenate and calibrate sample features
        sample_feats = torch.cat([img_feat, text_feat], dim=-1)
        with torch.no_grad():
            sample_feats = self.model.mlp(sample_feats).squeeze(0)
        sample_feats = F.normalize(sample_feats, dim=-1)
        
        # Compute cosine similarities and rank
        cos_sim = self.calib_data_feats @ sample_feats
        rank = torch.argsort(cos_sim, dim=0, descending=True).cpu().numpy()
        
        # Retrieve top-k training images
        top_idxs = rank[:top_k]
        retrieved = [self.coco_vis[i][0] for i in top_idxs]
        
        return query_img, sample_caption, retrieved


def main():
    args = parse_args()
    
    # Initialize demo
    demo_obj = AttributionDemo(
        args.checkpoint,
        args.data_dir,
        args.feature_dir,
    )
    
    # Create Gradio interface
    with gr.Blocks(title="FastGDA: Image Attribution Demo") as demo:
        gr.Markdown("""
        # FastGDA: Fast Gradient-based Data Attribution
        
        This demo shows image attribution on the MSCOCO dataset. 
        Enter a sample index to find the top-10 most influential training images.
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                idx_input = gr.Number(
                    label="Sample Index",
                    value=0,
                    precision=0,
                    minimum=0,
                    maximum=len(demo_obj.sample_captions) - 1
                )
                submit_btn = gr.Button("Find Attributions", variant="primary")
                top_k_slider = gr.Slider(
                    minimum=1,
                    maximum=50,
                    value=10,
                    step=1,
                    label="Number of Results"
                )
            
            with gr.Column(scale=2):
                query_img = gr.Image(label="Query Image", type="pil", height=256)
                caption_text = gr.Textbox(label="Caption", lines=3)
        
        gr.Markdown("### Top Attributed Training Images")
        gallery = gr.Gallery(
            label="Top Attributed Images",
            show_label=True,
            columns=5,
            rows=2,
            height=256 * 2
        )
        
        submit_btn.click(
            fn=demo_obj.predict,
            inputs=[idx_input, top_k_slider],
            outputs=[query_img, caption_text, gallery]
        )
    
    # Launch
    demo.launch(
        server_port=args.port
    )


if __name__ == "__main__":
    main()



