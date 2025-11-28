import os
import argparse
import torch
import numpy as np
from PIL import Image
from tqdm.auto import tqdm
from torchvision import transforms
from pycocotools.coco import COCO
from diffusers import AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer


def center_crop(image):
    width, height = image.size  # Get dimensions
    new_width = new_height = min(width, height)

    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = (width + new_width) / 2
    bottom = (height + new_height) / 2

    # Crop the center of the image
    image = image.crop((left, top, right, bottom))

    return image


def collate_fn(batch):
    """Custom collate function to properly handle list of captions"""
    images = []
    captions = []
    for img, caps in batch:
        images.append(img)
        captions.append(caps)
    
    # Stack images into a tensor
    images = torch.stack(images, dim=0)
    
    # Keep captions as a list of lists
    return images, captions


class COCODataset(torch.utils.data.Dataset):
    def __init__(self, path="data/coco", split="train", flip=False):
        dataType = f"{split}2017"
        annFile = os.path.join(path, "trainset", "annotations", f"captions_{dataType}.json")
        self.flip = flip
        self.imgdir = os.path.join(path, "trainset", dataType)
        self.coco = COCO(annFile)
        self.img_ids = list(self.coco.imgs.keys())
        self.captions = self.coco.imgToAnns
        self.preprocess = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __getitem__(self, idx):
        # get image
        i = self.img_ids[idx]
        img_dict = self.coco.loadImgs([i])[0]
        path = os.path.join(self.imgdir, img_dict["file_name"])

        image = Image.open(path).convert("RGB")
        im = center_crop(image).resize((128, 128))

        if self.flip:
            im = transforms.RandomHorizontalFlip(p=1)(im)

        # get first 5 captions for each image
        anns = self.captions[i]
        captions = [anns[j]["caption"] for j in range(5)]
        return self.preprocess(im), captions

    def __len__(self):
        return len(self.img_ids)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--latents_path", type=str, default="results/coco_train_latents.npy", help="output path for latents")
    parser.add_argument("--text_embeds_path", type=str, default="results/coco_train_text_embeds.npy", help="output path for text embeddings")
    parser.add_argument("--dataroot", type=str, default="../../data/coco", help="data root")
    parser.add_argument("--split", type=str, default="train", help="train or val")
    parser.add_argument("--batch_size", type=int, default=256, help="batch size")
    args = parser.parse_args()

    # Disable gradient computation
    torch.set_grad_enabled(False)

    # load vae
    # model_id = "stabilityai/stable-diffusion-2"
    model_id = "Manojb/stable-diffusion-2-1-base" # since stable-diffusion-2 is not available on HuggingFace, we use this backup model card instead
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae").cuda()
    vae.requires_grad_(False)
    
    # load text encoder and tokenizer
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder").cuda()
    text_encoder.requires_grad_(False)

    # prepare datasets
    coco_dataset_noflip = COCODataset(path=args.dataroot, split=args.split, flip=False)
    coco_dataset_flip = COCODataset(path=args.dataroot, split=args.split, flip=True)
    coco_dataset = torch.utils.data.ConcatDataset([coco_dataset_noflip, coco_dataset_flip])
    coco_loader = torch.utils.data.DataLoader(coco_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)

    # First loop: collect image latents
    print("Generating image latents...")
    all_latents = []
    for clean_image, captions in tqdm(coco_loader):
        latent = vae.encode(clean_image.cuda()).latent_dist.sample()
        latent = latent * vae.config.scaling_factor
        all_latents.append(latent.cpu().numpy())

    all_latents = np.concatenate(all_latents, axis=0)
    print("latents shape:", all_latents.shape)
    
    # Second loop: collect text embeddings
    # no need to flip since text embeddings are not affected by flipping
    coco_loader = torch.utils.data.DataLoader(coco_dataset_noflip, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
    print("Generating text embeddings...")
    all_text_embeds = []
    for clean_image, captions_batch in tqdm(coco_loader):
        # captions_batch is a list of length batch_size, each containing a list of 5 captions
        # Flatten to get all captions: batch_size * 5 captions
        num_images = len(captions_batch)  # batch_size
        num_captions_per_image = 5
        
        # Flatten the nested list structure
        all_captions = []
        for i in range(num_images):
            for j in range(num_captions_per_image):
                all_captions.append(captions_batch[i][j])
        
        # Tokenize all captions at once
        text_inputs = tokenizer(
            all_captions,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeds = text_encoder(text_inputs.input_ids.cuda())[0]
        
        # Reshape from (num_images * 5, 77, 1024) to (num_images, 5, 77, 1024)
        text_embeds = text_embeds.cpu().numpy()
        text_embeds = text_embeds.reshape(num_images, num_captions_per_image, 77, -1)
        all_text_embeds.append(text_embeds)

    all_text_embeds = np.concatenate(all_text_embeds, axis=0)
    print("text embeddings shape:", all_text_embeds.shape)

    # save latents and text embeddings
    os.makedirs(os.path.dirname(args.latents_path), exist_ok=True)
    np.save(args.latents_path, all_latents)
    
    os.makedirs(os.path.dirname(args.text_embeds_path), exist_ok=True)
    np.save(args.text_embeds_path, all_text_embeds)
