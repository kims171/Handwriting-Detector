# Handwriting-Detector

Handwriting Detector integrated with CRNN & ViT. A project combining a Vision Transformer (ViT) based detector + CNN feature extractor and a CRNN-based recognizer for handwritten text. This README documents the repo layout, environment requirements, and step-by-step instructions for training ViT and CRNN models, evaluating ViT on test set, and serving the models via a Gradio web interface.

---

## 1. Repository structure
- `data/` — Data preprocessing scripts and augmentation pipeline
- `utilities/` — helper scripts for defining optimizers, schedulers, and evaluation metrics
- `src/` — Core source code
  - `src/main.py` — entry point for training and evaluation
  - `src/train.py` — training logic and functions
  - `src/model.py` — model definitions (ViT wrappers, CRNN model, heads)
  - `src/cnn_feature_extractor.py` — Modified ResNet-18 backbone
- `checkpoints/` — Saved checkpoints and exported models (gitignored)
- `Handwriting_Recognition_Pipeline.ipynb` - Jupyter notebook for integrating ViT and CRNN for end-to-end handwriting recognition
- `requirements.py` — Project dependencies
- `README.md` — this file

---

## 2. requirements.py
```python
# requirements.py
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
```

Install with pip (example):

```bash
# use a virtual environment
python -m venv venv
source venv/bin/activate # on Linux/Mac
# or on Windows:
# venv/Scripts/activate

# Convert to requirements.txt:
python -c "import requirements; print('\\n'.join(requirements.REQUIREMENTS))" > requirements.txt
pip install -r requirements.txt
```

Or directly:

```bash
pip install torch torchvision timm transformers numpy pandas opencv-python Pillow albumentations einops scikit-learn editdistance python-Levenshtein tensorboard wandb gradio pyyaml
```

---

## 3. Train the ViT

Example training command

```bash
python src/main.py --ckpt_dir checkpoints/ --num_workers 4 --optimizer sam > train_v2.log 2>&1 &
```

After training is finished, `best_vit.pt` and `last_vit.pt` will be saved under the `checkpoints/` directory. You can evaluate the model on a test set using:

```bash
python utilities/evaluate_test.py --ckpt checkpoints/full_run_v2/best.pt --vocab checkpoints/full_run_v2/vocab.json
 ```

The CER and WER will be printed to the console. `best_vit.pt` is the best checkpoint based on validation CER. `last_vit.pt` is the last checkpoint saved during training.

## 4. Command Line Arguments

### 4.1 Data Configuration

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--dataset_name` | `str` | `"Teklia/IAM-line"` | Dataset repository or local path (e.g. HuggingFace dataset). |
| `--target_w` | `int` | `512` | Input line image target width (px). |
| `--target_h` | `int` | `64` | Input line image target height (px). |
| `--max_label_len` | `int` | `None` | Filters out training lines exceeding this target text length to stay comfortably within model CTC capacity ($L=128$). |

### 4.2 Model Architecture

*Defaults match Section 3.2 implementation details of the paper.*

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--dim` | `int` | `768` | Transformer hidden embedding dimension. |
| `--num_heads` | `int` | `6` | Number of multi-head attention heads. |
| `--mlp_hidden_dim` | `int` | `3072` | Hidden dimension of the feed-forward (MLP) layers. |
| `--num_layers` | `int` | `4` | Number of Transformer encoder layers. |
| `--mask_ratio` | `float` | `0.4` | Proportion of input patch sequence masked out during pretraining/training. |
| `--span_length` | `int` | `8` | Consecutive patch span length for span masking. |

### 4.3 Optimization & Training

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--batch_size` | `int` | `128` | Per-step batch size. |
| `--max_lr` | `float` | `1e-3` | Peak learning rate following linear warmup. |
| `--weight_decay` | `float` | `0.5` | Weight decay factor. |
| `--warmup_iters` | `int` | `1000` | Number of linear learning rate warmup steps. |
| `--total_iters` | `int` | `100000` | Total training iterations. |
| `--optimizer` | `str` | `"sam"` | Optimizer choice: `"sam"` (Sharpness-Aware Minimization) or `"adamw"`. |
| `--rho` | `float` | `0.05` | SAM neighborhood size ($\rho$). |
| `--grad_clip_norm` | `float` | `1.0` | Max gradient norm clipping threshold (applied on both SAM passes). Set `0` to disable. |
| `--ema_decay` | `float` | `0.9999` | Exponential Moving Average decay rate. |
| `--ema_warmup_steps` | `int` | `2000` | Steps over which EMA decay ramps linearly from `0.0` to `ema_decay` (prevents initial weights from dominating early validation metrics). |

### 4.4 Logging & Checkpointing

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--eval_every` | `int` | `1000` | Step interval for validation evaluation runs. |
| `--log_every` | `int` | `100` | Step interval for stdout logging. |
| `--ckpt_dir` | `str` | `"checkpoints"` | Output directory for saved checkpoints. |
| `--num_workers` | `int` | `4` | Number of `DataLoader` worker processes. |
| `--seed` | `int` | `0` | Global seed for reproducibility. |
| `--resume` | `str` | `None` | Path to checkpoint file (e.g., `checkpoints/last.pt`) to resume execution. |

---

## 5. Train the CRNN

Simply run the `Handwriting_Recognition_Pipeline.ipynb` end-to-end to train the CRNN model on the IAM lines dataset. However, before that, ensure that the `checkpoints/` directory contains the best ViT checkpoint (`best_vit.pt`) and the corresponding `vocab.json` file. If it does not exist, go to step 3 and train the ViT model first.

## 6. Gradio Demo

After running the `Handwriting_Recognition_Pipeline.ipynb` notebook end-to-end, you can access the Gradio interface by clicking the link in the last cell.
