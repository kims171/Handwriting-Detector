REQUIREMENTS = [
    "torch>=1.12.0",
    "torchvision>=0.13.0",
    "timm>=0.6.0",               # for ViT backbones and helpers
    "transformers>=4.30.0",      # optional if using HuggingFace ViT
    "numpy",
    "pandas",
    "opencv-python",
    "Pillow",
    "albumentations",
    "einops",
    "scikit-learn",
    "editdistance",              # evaluation utilities (optional)
    "python-Levenshtein",        # faster string distance (optional)
    "tensorboard",
    "wandb",                     # optional experiment logging
    "gradio",
    "pyyaml"
]