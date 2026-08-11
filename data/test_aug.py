"""
Run this locally, in the same directory as augmentations.py, where you have
the Teklia/IAM-line dataset loaded.

For each of a few real IAM lines, saves:
  - the original (resized) image
  - each individual augmentation applied to it
  - a few draws of the full combined pipeline (p=0.5 each)

so you can flip through aug_samples_real/ and check things look sane before
wiring this into the Dataset class for training.
"""

import os
import random
import numpy as np
from PIL import Image
from datasets import load_dataset

from data.augmentations import (
    random_affine,
    random_erosion,
    random_dilation,
    random_color_jitter,
    random_elastic_distortion,
    apply_augmentations,
)


def resize_direct(image, target_w=512, target_h=64):
    image = image.convert("L")
    return image.resize((target_w, target_h), Image.BILINEAR)


def main(n_samples=5, seed=0):
    random.seed(seed)
    np.random.seed(seed)

    ds = load_dataset("Teklia/IAM-line")
    train = ds["train"]

    out_dir = "aug_samples_real"
    os.makedirs(out_dir, exist_ok=True)

    # pick a few random real lines rather than just the first N,
    # so we see some variety in stroke thickness / writer style
    indices = random.sample(range(len(train)), n_samples)

    individual_augs = [
        ("erosion", random_erosion),
        ("dilation", random_dilation),
        ("affine", random_affine),
        ("color_jitter", random_color_jitter),
        ("elastic", random_elastic_distortion),
    ]

    for i, idx in enumerate(indices):
        example = train[idx]
        text = example["text"]
        img = resize_direct(example["image"])
        img_np = np.array(img)

        sample_dir = os.path.join(out_dir, f"sample_{i}_idx{idx}")
        os.makedirs(sample_dir, exist_ok=True)

        Image.fromarray(img_np).save(os.path.join(sample_dir, "00_original.png"))
        with open(os.path.join(sample_dir, "text.txt"), "w") as f:
            f.write(text)

        print(f"\nSample {i} (dataset idx {idx}): {text!r}")

        # individual augmentations, deterministic-ish per sample
        for name, fn in individual_augs:
            out = fn(img_np.copy())
            Image.fromarray(out).save(os.path.join(sample_dir, f"{name}.png"))

        # a few draws of the combined pipeline
        for j in range(3):
            out, applied = apply_augmentations(img_np.copy(), p=0.5)
            Image.fromarray(out).save(os.path.join(sample_dir, f"combined_{j}.png"))
            print(f"  combined_{j}: applied={applied}")

    print(f"\nAll samples saved under ./{out_dir}/ — one subfolder per sample,")
    print("each with the original, every individual augmentation, and 3 combined draws.")


if __name__ == "__main__":
    main(n_samples=5)