import torch
import editdistance


def greedy_decode(log_probs, blank_idx, idx2char):
    """log_probs: [B, L, num_classes+1] (post log_softmax).
    Returns a list of B decoded strings.
    """
    pred_ids = log_probs.argmax(dim=-1)  # [B, L]
    B = pred_ids.shape[0]
    decoded = []

    for b in range(B):
        seq = pred_ids[b].tolist()

        # collapse consecutive repeats
        collapsed = []
        prev = None
        for idx in seq:
            if idx != prev:
                collapsed.append(idx)
            prev = idx

        # remove blanks
        chars = [idx2char[idx] for idx in collapsed if idx != blank_idx]
        decoded.append("".join(chars))

    return decoded


def compute_cer(pred, gt):
    """Character error rate for a single (pred, gt) string pair."""
    if len(gt) == 0:
        return 0.0 if len(pred) == 0 else 1.0
    return editdistance.eval(pred, gt) / len(gt)


def compute_wer(pred, gt):
    """Word error rate for a single (pred, gt) string pair."""
    pred_words = pred.split()
    gt_words = gt.split()
    if len(gt_words) == 0:
        return 0.0 if len(pred_words) == 0 else 1.0
    return editdistance.eval(pred_words, gt_words) / len(gt_words)


def evaluate_batch(log_probs, gt_strings, blank_idx, idx2char):
    """Returns (mean_cer, mean_wer, list_of_decoded_strings) for a batch.
    Note: for corpus-level CER/WER (matching how the paper likely aggregates
    across a full dataset), you should accumulate total edit distance and
    total gt length across ALL batches, then divide once at the end --
    NOT average per-batch CER values, which weights short and long lines
    equally regardless of their actual character count. This function
    returns per-batch means for quick sanity checks only."""
    preds = greedy_decode(log_probs, blank_idx, idx2char)

    cers = [compute_cer(p, g) for p, g in zip(preds, gt_strings)]
    wers = [compute_wer(p, g) for p, g in zip(preds, gt_strings)]

    mean_cer = sum(cers) / len(cers)
    mean_wer = sum(wers) / len(wers)
    return mean_cer, mean_wer, preds


class CorpusMetrics:
    """Accumulates total edit distance and total length across an entire
    dataset, for correct corpus-level CER/WER."""

    def __init__(self):
        self.total_char_edits = 0
        self.total_char_len = 0
        self.total_word_edits = 0
        self.total_word_len = 0

    def update(self, preds, gts):
        for pred, gt in zip(preds, gts):
            self.total_char_edits += editdistance.eval(pred, gt)
            self.total_char_len += len(gt)

            pred_words, gt_words = pred.split(), gt.split()
            self.total_word_edits += editdistance.eval(pred_words, gt_words)
            self.total_word_len += len(gt_words)

    def compute(self):
        cer = self.total_char_edits / max(self.total_char_len, 1)
        wer = self.total_word_edits / max(self.total_word_len, 1)
        return cer, wer


if __name__ == "__main__":
    # ---- Unit tests for the decode/collapse logic, using known sequences ----
    # build a tiny fake vocab: a=0, b=1, l=2, blank=3
    idx2char = {0: "a", 1: "b", 2: "l", 3: ""}  # blank maps to empty, but we filter it anyway
    blank_idx = 3

    test_cases = [
        # (raw predicted index sequence, expected decoded string, description)
        ([0, 0, 0], "a", "pure repeats collapse to one character"),
        ([0, 3, 0], "aa", "blank-separated repeat survives as two characters"),
        ([2, 2, 3, 2, 2], "ll", "double-l word: 'll' needs a blank between the two l's"),
        ([3, 3, 0, 1, 3, 3], "ab", "leading/trailing blanks are stripped"),
        ([3, 3, 3], "", "all-blank sequence decodes to empty string"),
    ]

    print("---- Greedy decode unit tests ----")
    for seq, expected, desc in test_cases:
        log_probs = torch.zeros(1, len(seq), 4)
        for t, idx in enumerate(seq):
            log_probs[0, t, idx] = 100.0  # force argmax to pick this index
        decoded = greedy_decode(log_probs, blank_idx, idx2char)[0]
        status = "PASS" if decoded == expected else "FAIL"
        print(f"[{status}] {desc}: seq={seq} -> decoded={decoded!r} (expected {expected!r})")
        assert decoded == expected, f"Decode mismatch: got {decoded!r}, expected {expected!r}"

    # ---- CER/WER sanity checks ----
    print("\n---- CER/WER unit tests ----")
    cases = [
        ("hello", "hello", 0.0, 0.0),
        ("hello", "hallo", 1 / 5, 1.0),  # 1 char edit / 5 chars; 1 word edit / 1 word
        ("", "hello", 1.0, 1.0),
        ("the quick fox", "the quick brown fox", None, None),  # just eyeball this one
    ]
    for pred, gt, exp_cer, exp_wer in cases:
        cer = compute_cer(pred, gt)
        wer = compute_wer(pred, gt)
        print(f"pred={pred!r} gt={gt!r} -> CER={cer:.4f} WER={wer:.4f}"
              + (f" (expected CER={exp_cer}, WER={exp_wer})" if exp_cer is not None else ""))
        if exp_cer is not None:
            assert abs(cer - exp_cer) < 1e-6
            assert abs(wer - exp_wer) < 1e-6

    # ---- Corpus-level vs per-batch averaging: demonstrate why they differ ----
    print("\n---- Corpus-level vs naive per-batch averaging ----")
    preds = ["a", "the quick brown fox jumps over"]
    gts = ["b", "the quick brown fox jumps over"]  # one short wrong line, one long correct line

    naive_mean_cer = sum(compute_cer(p, g) for p, g in zip(preds, gts)) / len(preds)
    print(f"Naive per-sample mean CER: {naive_mean_cer:.4f} (treats both lines equally)")

    corpus = CorpusMetrics()
    corpus.update(preds, gts)
    corpus_cer, _ = corpus.compute()
    print(f"Corpus-level CER: {corpus_cer:.4f} (correctly weights by character count -- "
          f"the long correct line dominates, since CER should reflect total characters "
          f"gotten wrong across the WHOLE test set, not per-line average)")

    # ---- End-to-end test against the actual (untrained) model ----
    print("\n---- End-to-end test with the real model (untrained, expect garbage output) ----")
    from src.cnn_feature_extractor import ResNet18FeatureExtractor
    from utils.positional_and_masking import SinusoidalPositionalEmbedding, SpanMasking
    from utils.transformer_encoder import TransformerEncoder, CTCHead
    import torch.nn as nn

    class HTRVT(nn.Module):
        def __init__(self, num_classes=79, dim=768, num_heads=6, mlp_hidden_dim=3072,
                     num_layers=4, max_len=128, mask_ratio=0.4, span_length=8):
            super().__init__()
            self.cnn = ResNet18FeatureExtractor(out_channels=dim)
            self.pos_embed = SinusoidalPositionalEmbedding(max_len=max_len, dim=dim)
            self.span_mask = SpanMasking(dim=dim, mask_ratio=mask_ratio, span_length=span_length)
            self.encoder = TransformerEncoder(dim=dim, num_heads=num_heads,
                                               mlp_hidden_dim=mlp_hidden_dim, num_layers=num_layers)
            self.head = CTCHead(dim=dim, num_classes=num_classes)

        def forward(self, images):
            tokens = self.cnn(images)
            tokens = self.pos_embed(tokens)
            tokens, _ = self.span_mask(tokens)
            encoded = self.encoder(tokens)
            return self.head(encoded)

    # tiny fake char vocab matching build_vocab's convention (blank = last index)
    fake_chars = sorted(set("the quick brown fox jumps over lazy dog"))
    char2idx = {c: i for i, c in enumerate(fake_chars)}
    idx2char_real = {i: c for i, c in enumerate(fake_chars)}
    blank_idx_real = len(fake_chars)

    model = HTRVT(num_classes=len(fake_chars))
    model.eval()

    images = torch.randn(2, 1, 64, 512)
    with torch.no_grad():
        log_probs = model(images)

    gt_strings = ["the quick fox", "lazy dog jumps"]
    mean_cer, mean_wer, preds = evaluate_batch(log_probs, gt_strings, blank_idx_real, idx2char_real)

    print("Ground truth: ", gt_strings)
    print("Predictions (untrained, expect near-random garbage):", preds)
    print(f"Mean CER: {mean_cer:.4f}, Mean WER: {mean_wer:.4f}")
    print("(CER/WER near or above 1.0 is EXPECTED here -- the model hasn't been trained, "
          "this only confirms the decode+metric pipeline runs without errors)")
