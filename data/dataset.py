import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

import numpy as np
from PIL import Image

from ..data.augmentations import apply_augmentations

def resize_direct(image, target_w=512, target_h=64):
    image = image.convert("L")
    image = image.resize((target_w, target_h), Image.BILINEAR)
    return image

def collate_fn(batch):
    images, labels, label_lengths = zip(*batch)
    images = torch.stack(images)  # all same size due to fixed-resolution resize
    label_lengths = torch.tensor(label_lengths, dtype=torch.long)
    labels_concat = torch.cat(labels)  # CTC wants concatenated targets, not padded
    return images, labels_concat, label_lengths

def cycle_data(iterable):
    while True:
        for x in iterable:
            yield x

class IAMLineDataset(Dataset):
    def __init__(self, hf_dataset, char2idx, target_w=512, target_h=64, augment=False):
        self.dataset = hf_dataset
        self.char2idx = char2idx
        self.target_w = target_w
        self.target_h = target_h
        self.augment = augment
        self.to_tensor = T.ToTensor()  # scales to [0,1]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        image = resize_direct(example["image"], self.target_w, self.target_h)
        img_np = np.array(image)  # PIL -> numpy, uint8, (H, W)
        if self.augment:
            img_np, _ = apply_augmentations(img_np, p=0.5)
            image = Image.fromarray(img_np)
        image = self.to_tensor(image)  # shape [1, 64, 512]
        image = (image - 0.5) / 0.5    # normalize to [-1, 1]

        text = example["text"] # create label tensor
        label = torch.tensor([self.char2idx[c] for c in text if c in self.char2idx], dtype=torch.long)

        return image, label, len(label)