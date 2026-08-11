"""
IAM data pipeline for HTR-VT
"""
from functools import partial
from collections import Counter
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from datasets import load_dataset
from torchvision.transforms import ColorJitter

from data.augmentations import random_affine, random_erosion, random_dilation, random_elastic_distortion


# ---------- Vocab ----------
def build_vocab(dataset):
    charset = Counter()
    for example in dataset:
        charset.update(example["text"])
    chars = sorted(charset.keys())
    char2idx = {c: i for i, c in enumerate(chars)}
    idx2char = {i: c for i, c in enumerate(chars)}
    blank_idx = len(chars)
    return char2idx, idx2char, blank_idx


# ---------- Resize: aspect-preserving, capped, right-padded ----------
def resize_iam_style(image, target_w=512, target_h=64):
    image = image.convert("L")
    w, h = image.size
    new_w = min(int(w * target_h / h), target_w)
    new_w = max(new_w, 1)
    image = image.resize((new_w, target_h), Image.BILINEAR)

    canvas = Image.new("L", (target_w, target_h), color=255)  # white background
    canvas.paste(image, (0, 0))
    return canvas


# ---------- Dataset: returns RAW (unaugmented, unnormalized-beyond-[0,1]) tensors ----------
class IAMLineDataset(Dataset):
    def __init__(self, hf_dataset, char2idx, target_w=512, target_h=64, max_label_len=None):
        self.dataset = hf_dataset
        self.char2idx = char2idx
        self.target_w = target_w
        self.target_h = target_h

        # Optional: filter out lines whose label exceeds max_label_len.
        # The official myLoadDS supports this (mln/fmin args) -- useful for
        # keeping label length comfortably under your CTC sequence length L.
        if max_label_len is not None:
            keep = [i for i in range(len(hf_dataset)) if len(hf_dataset[i]["text"]) <= max_label_len]
            self.indices = keep
        else:
            self.indices = list(range(len(hf_dataset)))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        example = self.dataset[self.indices[idx]]
        image = resize_iam_style(example["image"], self.target_w, self.target_h)
        image_np = np.array(image, dtype=np.float32) / 255.0  # [0,1], matches img_as_float32 + /255.

        text = example["text"]
        missing = [c for c in text if c not in self.char2idx]
        if missing:
            print(f"WARNING idx={idx}: chars not in vocab: {missing}")
        label = [self.char2idx[c] for c in text if c in self.char2idx]

        # Return the raw [0,1] numpy image (not yet a normalized tensor) --
        # collate_fn_with_aug will do the uint8 round-trip for augmentation,
        # then renormalize, matching SameTrCollate's flow.
        return image_np, label


# ---------- Collate function: batch-level augmentation, matching SameTrCollate ----------

def collate_fn(batch, p=0.5, dila_ero_max_kernel=3, jitter_brightness=0.3,
                     jitter_contrast=0.3, jitter_saturation=0.0, jitter_hue=0.0):
        images, labels = zip(*batch)
        # images: tuple of numpy [0,1] float arrays, shape (H, W)

        # convert to uint8 PIL images for augmentation, matching SameTrCollate
        pil_images = [Image.fromarray(np.uint8(img * 255)) for img in images]

        # 1) random affine-style transform -- one coin flip for the whole batch
        if random.random() < p:
            pil_images = [Image.fromarray(random_affine(np.array(im))) for im in pil_images]

        # 2) erosion OR dilation -- one coin flip decides IF, another decides WHICH
        if random.random() < p:
            if random.random() < 0.5:
                pil_images = [Image.fromarray(random_erosion(np.array(im))) for im in pil_images]
            else:
                pil_images = [Image.fromarray(random_dilation(np.array(im))) for im in pil_images]

        # 3) color jitter -- one coin flip for the whole batch
        if random.random() < p:
            jitter = ColorJitter(jitter_brightness, jitter_contrast, jitter_saturation, jitter_hue)
            pil_images = [jitter(im) for im in pil_images]

        # 4) elastic distortion -- kept as our own addition per the paper's text;
        #    see note above about this not being confirmed in the official snippet
        if random.random() < p:
            pil_images = [Image.fromarray(random_elastic_distortion(np.array(im))) for im in pil_images]

        # back to normalized float tensors, [0,1], shape [B, 1, H, W]
        image_tensors = [torch.from_numpy(np.array(im, copy=True)) for im in pil_images]
        image_tensors = torch.stack(image_tensors).unsqueeze(1).float() / 255.0

        label_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)
        labels_concat = torch.cat([torch.tensor(l, dtype=torch.long) for l in labels])

        return image_tensors, labels_concat, label_lengths

def make_collate_fn(p=0.5, dila_ero_max_kernel=3, jitter_brightness=0.3,
                     jitter_contrast=0.3, jitter_saturation=0.0, jitter_hue=0.0):
    return partial(collate_fn, p=p, dila_ero_max_kernel=dila_ero_max_kernel, jitter_brightness=jitter_brightness,
                   jitter_contrast=jitter_contrast, jitter_saturation=jitter_saturation, jitter_hue=jitter_hue)

# ---------- Eval-time collate: no augmentation ----------

def collate_fn_eval(batch):
    images, labels = zip(*batch)
    image_tensors = torch.stack([torch.from_numpy(img) for img in images]).unsqueeze(1).float()
    label_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)
    labels_concat = torch.cat([torch.tensor(l, dtype=torch.long) for l in labels])
    return image_tensors, labels_concat, label_lengths
