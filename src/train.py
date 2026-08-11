"""
Training loop for HTR-VT.

Hyperparameters below match the paper's stated implementation details
(Section 3.2): batch size 128, AdamW inside SAM with weight decay 0.5,
max LR 1e-3, 1,000 warmup iterations, 100,000 total iterations, EMA decay
0.9999. Training is ITERATION-based, not epoch-based -- we cycle through the
DataLoader indefinitely rather than looping over epochs, matching the
official repo's cycle_data() helper we saw earlier.

This file assumes a SAM optimizer with the standard davda54/sam-style API:
    optimizer.first_step(zero_grad=True)
    optimizer.second_step(zero_grad=True)
and that optimizer.param_groups is a normal list of param-group dicts (true
for that implementation, since it wraps a base_optimizer and copies its
param_groups) -- so a standard LR-setting loop works directly on it. If your
SAM implementation's API differs, the two call sites are isolated below.
"""

# ---------- Full optimizer state save/load (handles SAM's nested base_optimizer) ----------
# SAM wraps a separate base_optimizer (e.g. AdamW) with its OWN independent
# .state dict, where the actual momentum buffers (exp_avg, exp_avg_sq) live.
# SAM's own (outer) .state only holds transient "e_w" perturbation values.
# Calling the outer optimizer.state_dict() alone silently drops all of
# AdamW's momentum -- confirmed this concretely: a fresh SAM optimizer after
# real training steps has real exp_avg values inside base_optimizer.state,
# but optimizer.state_dict()'s own state dict never contains them. This
# means every resume was cold-starting Adam's moment estimates while
# already at full LR, producing exactly the kind of burst-of-large-updates
# instability seen right at a resume point.
#
# If your SAM implementation doesn't use the attribute name "base_optimizer",
# adjust the getattr calls below to match.

def get_full_optimizer_state(optimizer):
    state = {"outer_state_dict": optimizer.state_dict()}
    base_opt = getattr(optimizer, "base_optimizer", None)
    if base_opt is not None:
        state["base_optimizer_state_dict"] = base_opt.state_dict()
    return state


def load_full_optimizer_state(optimizer, saved):
    base_opt = getattr(optimizer, "base_optimizer", None)

    if "outer_state_dict" not in saved:
        # OLD format: `saved` IS the raw optimizer.state_dict() output, from a
        # checkpoint saved before this fix existed (e.g. the original best.pt
        # from before we introduced get_full_optimizer_state). No separate
        # base_optimizer state was ever captured -- momentum cannot be
        # recovered, this is a real, unavoidable one-time cold start.
        print("WARNING: checkpoint uses the OLD optimizer-state format (pre-fix) -- "
              "Adam momentum will cold-start from this resume. This should only "
              "happen for checkpoints saved before the optimizer-state fix; any "
              "checkpoint saved AFTER this point will resume correctly.")
        optimizer.load_state_dict(saved)
        if base_opt is not None:
            # still apply the param_groups re-sync even on cold start, so at
            # least the LR-propagation bug doesn't ALSO bite on top of this
            base_opt.param_groups = optimizer.param_groups
        return

    optimizer.load_state_dict(saved["outer_state_dict"])
    if base_opt is not None and "base_optimizer_state_dict" in saved:
        base_opt.load_state_dict(saved["base_optimizer_state_dict"])
        # base_opt.load_state_dict() just rebuilt base_opt's OWN param_groups
        # list, breaking the shared-reference invariant SAM's __init__ sets up
        # (self.param_groups IS self.base_optimizer.param_groups, same object).
        # Confirmed concretely that without this line, set_lr()'s per-step
        # mutation of optimizer.param_groups silently stops reaching the real
        # AdamW update after any resume -- training would keep running at a
        # frozen, stale LR with no error or warning. Restore the shared
        # reference explicitly, matching what this SAM's own load_state_dict
        # override does for the non-base_optimizer_state_dict case.
        base_opt.param_groups = optimizer.param_groups
    elif base_opt is not None:
        print("WARNING: checkpoint has no base_optimizer_state_dict (likely saved "
              "before this fix) -- Adam momentum will cold-start from this resume.")


import math
import copy
import os
import torch
import torch.nn as nn

from utilities.ctc_decode_metrics import greedy_decode, CorpusMetrics


# ---------- LR schedule: warmup then cosine decay ----------

def warmup_cosine_lr(step, warmup_iters, total_iters, max_lr):
    if step < warmup_iters:
        return max_lr * step / max(warmup_iters, 1)
    progress = (step - warmup_iters) / max(total_iters - warmup_iters, 1)
    progress = min(progress, 1.0)
    return max_lr * 0.5 * (1 + math.cos(math.pi * progress))


def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group["lr"] = lr


# ---------- EMA ----------

class EMA:
    """Exponential moving average over a model's float parameters and
    buffers (so e.g. BatchNorm running stats get EMA'd too, matching common
    practice for EMA implementations like timm's ModelEmaV2).

    Uses a WARMUP schedule for the decay rate: ramps linearly from 0 up to
    target_decay over warmup_steps, rather than applying target_decay from
    step 1. Without this, decay=0.9999 gives an effective half-life of
    ln(2)/(1-0.9999) ~= 6931 steps -- confirmed concretely that this caused
    the EMA weights to still be ~74% dominated by random initialization at
    step 3000, producing near-garbage validation predictions (blank
    collapse) even while the live model was training completely normally
    underneath. The paper states decay=0.9999 as a flat constant with no
    mention of warmup, so this is a deviation from the literal text -- but
    a very standard, common one (this exact linear-ramp technique is used
    in most practical EMA implementations, e.g. timm's ModelEmaV2), and
    necessary for early-training evaluation to mean anything at all.
    """

    def __init__(self, model, decay=0.9999, warmup_steps=2000):
        self.target_decay = decay
        self.warmup_steps = warmup_steps
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    def get_decay(self, step):
        if self.warmup_steps <= 0:
            return self.target_decay
        return self.target_decay * min(1.0, step / self.warmup_steps)

    @torch.no_grad()
    def update(self, model, step):
        decay = self.get_decay(step)
        for shadow_p, model_p in zip(self.shadow.state_dict().values(),
                                      model.state_dict().values()):
            if shadow_p.dtype.is_floating_point:
                shadow_p.mul_(decay).add_(model_p, alpha=1 - decay)
            else:
                shadow_p.copy_(model_p)  # non-float buffers (e.g. counters): just copy


# ---------- Infinite data cycler (iteration-based training) ----------

def cycle(dataloader):
    while True:
        for batch in dataloader:
            yield batch


# ---------- One training step (SAM, two forward/backward passes) ----------

def train_step(model, optimizer, ctc_loss_fn, images, labels_concat, label_lengths, device,
                grad_clip_norm=1.0, loss_spike_threshold=20.0):
    images = images.to(device)
    labels_concat = labels_concat.to(device)
    label_lengths = label_lengths.to(device)

    B = images.shape[0]
    input_lengths = torch.full((B,), model.max_len, dtype=torch.long, device=device)

    # Sample the span mask ONCE, reuse across both SAM passes -- see the
    # docstring on HTRVT.sample_mask / SpanMasking.sample_mask for why.
    mask = model.sample_mask(B, device)

    def compute_loss():
        log_probs = model(images, mask=mask)
        log_probs_tm = log_probs.permute(1, 0, 2)  # CTCLoss wants [L, B, C]

        # Check if the active execution device is Apple Silicon MPS
        if str(device) == "mps":
            # Offload tensors to the host CPU safely to prevent NotImplementedError
            log_probs_cpu = log_probs_tm.to("cpu")
            labels_cpu = labels_concat.to("cpu")
            input_lengths_cpu = input_lengths.to("cpu")
            label_lengths_cpu = label_lengths.to("cpu")

            # Execute calculation on CPU and route output scalar back to MPS
            loss_cpu = ctc_loss_fn(log_probs_cpu, labels_cpu, input_lengths_cpu, label_lengths_cpu)
            loss = loss_cpu.to(device)
        else:
            # Default native execution pathway for CUDA or general CPU configurations
            loss = ctc_loss_fn(log_probs_tm, labels_concat, input_lengths, label_lengths)

        return loss

    # ---- SAM pass 1: compute gradient at current weights ----
    loss1 = compute_loss()
    if not torch.isfinite(loss1):
        return None  # caller should skip this step and log it
    loss1.backward()
    grad_norm1 = torch.nn.utils.clip_grad_norm_(
        model.parameters(), grad_clip_norm if grad_clip_norm else float("inf")
    )
    if loss1.item() > loss_spike_threshold:
        print(f"  [loss spike] pass1 loss={loss1.item():.2f} "
              f"grad_norm_pre_clip={grad_norm1:.2f}")
    optimizer.first_step(zero_grad=True)

    # ---- SAM pass 2: compute gradient at perturbed weights, step back ----
    loss2 = compute_loss()
    if not torch.isfinite(loss2):
        return None
    loss2.backward()
    grad_norm2 = torch.nn.utils.clip_grad_norm_(
        model.parameters(), grad_clip_norm if grad_clip_norm else float("inf")
    )
    if loss2.item() > loss_spike_threshold:
        print(f"  [loss spike] pass2 loss={loss2.item():.2f} "
              f"grad_norm_pre_clip={grad_norm2:.2f}")
    optimizer.second_step(zero_grad=True)

    return loss1.item()  # report the first-pass loss, matching common SAM logging convention


# ---------- Validation ----------

@torch.no_grad()
def validate(model, dataloader, blank_idx, idx2char, device, max_batches=None):
    model.eval()
    corpus = CorpusMetrics()

    for i, (images, labels_concat, label_lengths) in enumerate(dataloader):
        if max_batches is not None and i >= max_batches:
            break
        images = images.to(device)
        log_probs = model(images)  # eval mode -> span masking is a no-op automatically
        preds = greedy_decode(log_probs, blank_idx, idx2char)

        gts = []
        offset = 0
        for length in label_lengths.tolist():
            ids = labels_concat[offset:offset + length].tolist()
            gts.append("".join(idx2char[i] for i in ids))
            offset += length

        corpus.update(preds, gts)

    cer, wer = corpus.compute()
    model.train()
    return cer, wer


# ---------- Main training loop ----------

def train(model, optimizer, train_loader, val_loader, blank_idx, idx2char, device,
          total_iters=100_000, warmup_iters=1_000, max_lr=1e-3, ema_decay=0.9999,
          eval_every=1_000, ckpt_dir="checkpoints", log_every=100, resume_path=None,
          grad_clip_norm=1.0, ema_warmup_steps=2_000):

    os.makedirs(ckpt_dir, exist_ok=True)
    ctc_loss_fn = nn.CTCLoss(blank=blank_idx, zero_infinity=True)
    ema = EMA(model, decay=ema_decay, warmup_steps=ema_warmup_steps)
    data_iter = cycle(train_loader)

    start_step = 1
    best_val_cer = float("inf")

    if resume_path is not None and os.path.exists(resume_path):
        print(f"Resuming from {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        ema.shadow.load_state_dict(ckpt["ema_state_dict"])
        load_full_optimizer_state(optimizer, ckpt["optimizer_state_dict"])
        start_step = ckpt["step"] + 1
        best_val_cer = ckpt["best_val_cer"]
        print(f"Resumed at step {start_step}, best_val_cer so far: {best_val_cer:.4f}")
        print("NOTE: data iteration order is NOT restored exactly (the cycling "
              "DataLoader restarts fresh) -- this is a functional resume, not a "
              "bit-exact one. Fine in practice, just worth knowing.")

    running_loss = 0.0
    skipped_steps = 0

    for step in range(start_step, total_iters + 1):
        lr = warmup_cosine_lr(step, warmup_iters, total_iters, max_lr)
        set_lr(optimizer, lr)

        images, labels_concat, label_lengths = next(data_iter)
        loss = train_step(model, optimizer, ctc_loss_fn, images, labels_concat, label_lengths,
                           device, grad_clip_norm=grad_clip_norm)

        if loss is None:
            skipped_steps += 1
            if skipped_steps % 10 == 1:
                print(f"[step {step}] WARNING: non-finite loss, skipped step "
                      f"(total skipped so far: {skipped_steps})")
            continue

        ema.update(model, step)
        running_loss += loss

        if step % log_every == 0:
            avg_loss = running_loss / log_every
            print(f"[step {step}/{total_iters}] lr={lr:.6f} loss={avg_loss:.4f}")
            running_loss = 0.0

        if step % eval_every == 0:
            # Evaluate EMA weights (matches the paper's stated approach for
            # final reporting) AND live weights (diagnostic only -- added
            # after discovering EMA immaturity early in training made
            # validation numbers meaningless for thousands of steps despite
            # the live model training completely normally underneath).
            # Checkpointing/best-tracking still uses EMA, matching the
            # paper -- but printing both gives visibility to catch this
            # class of issue immediately instead of chasing it blind.
            val_cer, val_wer = validate(ema.shadow, val_loader, blank_idx, idx2char, device)
            ema.shadow.eval()  # validate() leaves it in .train() mode; ema.shadow should
                                # stay permanently in eval mode (it's averaged, never backprop-trained)
            live_cer, live_wer = validate(model, val_loader, blank_idx, idx2char, device)
            print(f"[step {step}] VAL(ema)  cer={val_cer:.4f} wer={val_wer:.4f} "
                  f"(best so far: {best_val_cer:.4f})")
            print(f"[step {step}] VAL(live) cer={live_cer:.4f} wer={live_wer:.4f}  "
                  f"[diagnostic only, not used for checkpointing]")

            improved = val_cer < best_val_cer
            if improved:
                best_val_cer = val_cer

            # build checkpoint AFTER best_val_cer is up to date, so last.pt
            # never saves a stale best value (this bit us in testing: saving
            # the dict before this update meant a resumed run could think a
            # worse checkpoint was "best" and overwrite a genuinely better one)
            checkpoint = {
                "step": step,
                "model_state_dict": model.state_dict(),
                "ema_state_dict": ema.shadow.state_dict(),
                "optimizer_state_dict": get_full_optimizer_state(optimizer),
                "best_val_cer": best_val_cer,
            }

            torch.save(checkpoint, os.path.join(ckpt_dir, "last_vit.pt"))

            if improved:
                torch.save(checkpoint, os.path.join(ckpt_dir, "best_vit.pt"))
                print(f"[step {step}] New best checkpoint saved (val_cer={best_val_cer:.4f})")

    print(f"\nTraining complete. Total skipped steps due to non-finite loss: {skipped_steps}")
    return best_val_cer
