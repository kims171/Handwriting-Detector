"""
Evaluate a trained checkpoint on the IAM test set, reporting corpus-level
CER/WER (accumulated over the whole set, not naively averaged per-batch --
see CorpusMetrics in ctc_decode_metrics.py for why that distinction matters).

Usage:
    python evaluate.py --ckpt checkpoints/full_run_v2/best.pt \
                        --vocab checkpoints/full_run_v2/vocab.json
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
import time

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset

from data.data_pipeline import IAMLineDataset, collate_fn_eval
from src.model import HTRVT
from utilities.ctc_decode_metrics import greedy_decode, CorpusMetrics


def evaluate(model, dataloader, blank_idx, idx2char, device, save_predictions_path=None):
    model.eval()
    corpus = CorpusMetrics()
    all_records = []

    with torch.no_grad():
        for images, labels_concat, label_lengths in dataloader:
            images = images.to(device)
            log_probs = model(images)
            preds = greedy_decode(log_probs, blank_idx, idx2char)

            gts = []
            offset = 0
            for length in label_lengths.tolist():
                ids = labels_concat[offset:offset + length].tolist()
                gts.append("".join(idx2char[i] for i in ids))
                offset += length

            corpus.update(preds, gts)

            if save_predictions_path is not None:
                all_records.extend(zip(preds, gts))

    if save_predictions_path is not None:
        with open(save_predictions_path, "w") as f:
            for pred, gt in all_records:
                f.write(f"GT:   {gt}\nPRED: {pred}\n\n")
        print(f"Saved {len(all_records)} predictions to {save_predictions_path}")

    cer, wer = corpus.compute()
    return cer, wer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--vocab", type=str, required=True)
    p.add_argument("--dataset_name", type=str, default="Teklia/IAM-line")
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--save_predictions", type=str, default=None,
                    help="Optional path to dump all (prediction, ground_truth) pairs "
                         "for qualitative review, e.g. predictions_test.txt")
    p.add_argument("--dim", type=int, default=768)
    p.add_argument("--num_heads", type=int, default=6)
    p.add_argument("--mlp_hidden_dim", type=int, default=3072)
    p.add_argument("--num_layers", type=int, default=4)
    args = p.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    with open(args.vocab) as f:
        vocab = json.load(f)
    char2idx = vocab["char2idx"]
    blank_idx = vocab["blank_idx"]
    idx2char = {idx: ch for ch, idx in char2idx.items()}
    print(f"Vocab size: {len(char2idx)}, blank_idx={blank_idx}")

    print(f"\nLoading {args.dataset_name}, split={args.split}")
    ds = load_dataset(args.dataset_name)
    test_ds = IAMLineDataset(ds[args.split], char2idx)
    print(f"{args.split}: {len(test_ds)} examples")

    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn_eval, num_workers=args.num_workers,
    )

    model = HTRVT(num_classes=len(char2idx), dim=args.dim, num_heads=args.num_heads,
                  mlp_hidden_dim=args.mlp_hidden_dim, num_layers=args.num_layers, max_len=128).to(device)

    ckpt = torch.load(args.ckpt, map_location=device)
    print(f"\nLoaded checkpoint: {args.ckpt}")
    print(f"  step: {ckpt.get('step', '?')}, best_val_cer (at checkpoint time): "
          f"{ckpt.get('best_val_cer', '?')}")

    # ---- EMA weights: the paper's methodology for final reporting ----
    model.load_state_dict(ckpt["ema_state_dict"])
    t0 = time.time()
    ema_cer, ema_wer = evaluate(model, test_loader, blank_idx, idx2char, device,
                                 save_predictions_path=args.save_predictions)
    ema_time = time.time() - t0

    # ---- live weights: diagnostic cross-check, given this project's EMA history ----
    model.load_state_dict(ckpt["model_state_dict"])
    t0 = time.time()
    live_cer, live_wer = evaluate(model, test_loader, blank_idx, idx2char, device)
    live_time = time.time() - t0

    print("\n" + "=" * 60)
    print(f"TEST SET RESULTS ({args.split}, {len(test_ds)} examples)")
    print("=" * 60)
    print(f"EMA  weights: CER={ema_cer:.4f}  WER={ema_wer:.4f}   ({ema_time:.1f}s)")
    print(f"Live weights: CER={live_cer:.4f}  WER={live_wer:.4f}   ({live_time:.1f}s)")
    print()
    print("Paper's reported IAM test numbers (Table 4): CER=4.7 (0.047), WER=14.9 (0.149)")

    gap = abs(ema_cer - live_cer)
    if gap > 0.05:
        print(f"\nNOTE: EMA and live CER differ by {gap:.4f} -- larger gap than expected "
              f"if training ran well past the EMA warmup window. Worth checking training "
              f"ran long enough, or investigating further before trusting either number blindly.")


if __name__ == "__main__":
    main()
