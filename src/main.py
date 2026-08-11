"""
Main training entry point for HTR-VT on IAM.
"""

import sys
import os

# Get the absolute path of the MAI202 root folder
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Insert it at index 0 so Python checks it FIRST before anything else
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import argparse
import json
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset

from data.data_pipeline import IAMLineDataset, build_vocab, make_collate_fn, collate_fn_eval
from src.model import HTRVT
from src.train import train
from utilities.plain_optimizer_adapter import PlainOptimizerAdapter
from utilities.optim_utils import build_optimizer_param_groups
from utilities.sam import SAM

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def parse_args():
    p = argparse.ArgumentParser()

    # data
    p.add_argument("--dataset_name", type=str, default="Teklia/IAM-line")
    p.add_argument("--target_w", type=int, default=512)
    p.add_argument("--target_h", type=int, default=64)
    p.add_argument("--max_label_len", type=int, default=None,
                    help="Optionally filter out training lines with labels longer than this. "
                         "Useful for keeping label length comfortably under the model's L=128 "
                         "CTC sequence length. Left unset by default -- check the printed "
                         "label-length-vs-L diagnostic at startup before deciding whether to set this.")

    # model (defaults match paper's Section 3.2 implementation details)
    p.add_argument("--dim", type=int, default=768)
    p.add_argument("--num_heads", type=int, default=6)
    p.add_argument("--mlp_hidden_dim", type=int, default=3072)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--mask_ratio", type=float, default=0.4)
    p.add_argument("--span_length", type=int, default=8)

    # optimization (defaults match paper's Section 3.2 implementation details)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--max_lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.5)
    p.add_argument("--warmup_iters", type=int, default=1_000)
    p.add_argument("--total_iters", type=int, default=100_000)
    p.add_argument("--ema_decay", type=float, default=0.9999)
    p.add_argument("--rho", type=float, default=0.05,
                    help="SAM's neighborhood size. NOT specified in the paper text -- "
                         "this is the standard default from the original SAM paper, unconfirmed "
                         "against HTR-VT specifically.")
    p.add_argument("--grad_clip_norm", type=float, default=1.0,
                    help="Max gradient norm for clipping (applied on both SAM passes). "
                         "NOT specified in the paper text -- added after observing a real "
                         "training instability/collapse without it. Set to 0 to disable.")
    p.add_argument("--optimizer", type=str, default="sam", choices=["sam", "adamw"],
                    help="'adamw' bypasses SAM entirely (via PlainOptimizerAdapter), for "
                         "isolating whether SAM specifically is contributing to an observed "
                         "training instability. Added after a real blank-collapse incident "
                         "to allow a controlled comparison.")
    p.add_argument("--ema_warmup_steps", type=int, default=2_000,
                    help="EMA decay ramps linearly from 0 to ema_decay over this many steps, "
                         "instead of applying full decay from step 1. NOT in the paper text -- "
                         "added after confirming EMA weights were still ~74% dominated by random "
                         "init at step 3000 with a flat decay=0.9999, causing validation to look "
                         "like a collapsed model even though the live model was training fine.")

    # logging / checkpointing
    p.add_argument("--eval_every", type=int, default=1_000)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--ckpt_dir", type=str, default="checkpoints")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", type=str, default=None,
                    help="Path to a checkpoint (e.g. checkpoints/last.pt) to resume from.")

    return p.parse_args()

def run(args):
    set_seed(args.seed)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")
    if device.type == "cpu":
        print("WARNING: no CUDA or MPS device found. Training HTR-VT at full size on CPU "
              "will be extremely slow.")

    # ---- Data ----
    print(f"\nLoading dataset: {args.dataset_name}")
    ds = load_dataset(args.dataset_name)
    for split in ds:
        print(f"  {split}: {len(ds[split])} examples")

    char2idx, idx2char, blank_idx = build_vocab(ds["train"])
    print(f"\nVocab size (excluding blank): {len(char2idx)}, blank_idx={blank_idx}")

    # save vocab now -- needed later for inference/evaluation without
    # re-deriving it from the full training set every time
    vocab_path = os.path.join(args.ckpt_dir, "vocab.json")
    with open(vocab_path, "w") as f:
        json.dump({"char2idx": char2idx, "blank_idx": blank_idx}, f)
    print(f"Saved vocab to {vocab_path}")

    train_ds = IAMLineDataset(ds["train"], char2idx, args.target_w, args.target_h,
                               max_label_len=args.max_label_len)
    val_split = "validation" if "validation" in ds else "val"
    val_ds = IAMLineDataset(ds[val_split], char2idx, args.target_w, args.target_h)

    L = args.target_w // 4  # matches the CNN's stem-only width downsampling (4x)
    train_label_lens = [len(ds["train"][i]["text"]) for i in range(len(ds["train"]))]
    max_label_len_seen = max(train_label_lens)
    print(f"\nCTC sequence length L={L}. Longest training label: {max_label_len_seen} characters.")
    if max_label_len_seen > L:
        print(f"WARNING: at least one training label ({max_label_len_seen} chars) exceeds L={L}. "
              f"CTC cannot align these -- either set --max_label_len to filter them out, or this "
              f"will produce infinite/NaN loss on those specific examples (zero_infinity=True in "
              f"the loss will suppress the crash but silently drops their gradient contribution).")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=make_collate_fn(), num_workers=args.num_workers,
        drop_last=True, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn_eval, num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # ---- Model ----
    model = HTRVT(
        num_classes=len(char2idx), dim=args.dim, num_heads=args.num_heads,
        mlp_hidden_dim=args.mlp_hidden_dim, num_layers=args.num_layers,
        max_len=L, mask_ratio=args.mask_ratio, span_length=args.span_length,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {n_params:,} ({n_params/1e6:.2f}M)")

    # ---- Optimizer ----
    param_groups, decay_params, no_decay_params = build_optimizer_param_groups(model, args.weight_decay)
    print(f"Optimizer param groups: {sum(p.numel() for p in decay_params):,} decayed, "
          f"{sum(p.numel() for p in no_decay_params):,} not decayed")

    if args.optimizer == "sam":
        optimizer = SAM(param_groups, torch.optim.AdamW, rho=args.rho, lr=args.max_lr)
    else:
        base_optimizer = torch.optim.AdamW(param_groups, lr=args.max_lr)
        optimizer = PlainOptimizerAdapter(base_optimizer)
    print(f"Optimizer: {args.optimizer}")

    # ---- Train ----
    print(f"\nStarting training: {args.total_iters} iterations, "
          f"batch size {args.batch_size}, eval every {args.eval_every}\n")

    best_val_cer = train(
        model, optimizer, train_loader, val_loader, blank_idx, idx2char, device,
        total_iters=args.total_iters, warmup_iters=args.warmup_iters, max_lr=args.max_lr,
        ema_decay=args.ema_decay, eval_every=args.eval_every, ckpt_dir=args.ckpt_dir,
        log_every=args.log_every, resume_path=args.resume,
        grad_clip_norm=(args.grad_clip_norm if args.grad_clip_norm > 0 else None),
        ema_warmup_steps=args.ema_warmup_steps,
    )

    print(f"\nDone. Best validation CER: {best_val_cer:.4f}")
    return best_val_cer

def main():
    args = parse_args()
    run(args)

if __name__ == "__main__":
    main()
