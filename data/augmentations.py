"""
Augmentation pipeline for random transformation, erosion, dilation, color jitter, and elastic distortion.
We set the probability of using each data augmentation to 0.5, and they can be
combined with each other.

Each function takes and returns a numpy uint8 array, shape (H, W), grayscale,
values in [0, 255], background ~255 (white), ink ~0 (dark).
"""

import numpy as np
import cv2
import random
from scipy.ndimage import gaussian_filter


# ---------- 1. Random affine transform ----------

def random_affine(img, max_rotate=3, max_translate=0.02, max_scale=0.05, max_shear=2):
    """Small random rotation/translation/scale/shear. Kept mild since line images
    are sensitive to distortion (unlike natural images)."""
    h, w = img.shape

    angle = random.uniform(-max_rotate, max_rotate)
    scale = 1.0 + random.uniform(-max_scale, max_scale)
    shear = random.uniform(-max_shear, max_shear)
    tx = random.uniform(-max_translate, max_translate) * w
    ty = random.uniform(-max_translate, max_translate) * h

    # rotation + scale about center
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    # add shear
    M[0, 1] += np.tan(np.radians(shear))
    # add translation
    M[0, 2] += tx
    M[1, 2] += ty

    out = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    return out


# ---------- 2. Erosion ----------

def random_erosion(img, kernel_size=2):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    out = cv2.dilate(img, kernel, iterations=1)  # dilate bright pixels -> thinner dark strokes
    return out


# ---------- 3. Dilation ----------

def random_dilation(img, kernel_range=(2, 3)):
    """Dilation of ink = erosion of the raw grayscale array (thickens dark strokes)."""
    k = random.choice(range(kernel_range[0], kernel_range[1] + 1))
    kernel = np.ones((k, k), np.uint8)
    out = cv2.erode(img, kernel, iterations=1)  # erode bright pixels -> thicker dark strokes
    return out


# ---------- 4. Color jitter (brightness/contrast, grayscale-appropriate) ----------

def random_color_jitter(img, brightness_range=0.3, contrast_range=0.3):
    brightness = 1.0 + random.uniform(-brightness_range, brightness_range)
    contrast = 1.0 + random.uniform(-contrast_range, contrast_range)

    img_f = img.astype(np.float32)
    mean = img_f.mean()
    img_f = (img_f - mean) * contrast + mean  # contrast around image mean
    img_f = img_f * brightness

    out = np.clip(img_f, 0, 255).astype(np.uint8)
    return out


# ---------- 5. Elastic distortion ----------

def random_elastic_distortion(img, alpha=34, sigma=4):
    """Classic Simard et al. elastic distortion, standard for handwriting augmentation."""
    h, w = img.shape
    dx = gaussian_filter((np.random.rand(h, w) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((np.random.rand(h, w) * 2 - 1), sigma) * alpha

    x, y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (x + dx).astype(np.float32)
    map_y = (y + dy).astype(np.float32)

    out = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    return out


# ---------- Compose: each applied independently w.p. 0.5 ----------

AUGMENTATIONS = [
    random_affine,
    random_erosion,
    random_dilation,
    random_color_jitter,
    random_elastic_distortion,
]


def apply_augmentations(img, p=0.5, augmentations=AUGMENTATIONS):
    """img: numpy uint8 array (H, W). Applies each augmentation independently
    with probability p, in a fixed order (order matters slightly for geometric
    vs. photometric ops)."""
    out = img.copy()
    applied = []
    for aug_fn in augmentations:
        if random.random() < p:
            out = aug_fn(out)
            applied.append(aug_fn.__name__)
    return out, applied
