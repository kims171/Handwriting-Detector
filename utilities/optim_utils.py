"""
Standard parameter-group split for AdamW-style weight decay: apply decay
only to weight matrices (2D+ params), NOT to biases or normalization
parameters (LayerNorm/BatchNorm weight & bias, both 1D).
"""

def build_optimizer_param_groups(model, weight_decay):
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # 1D params: biases, LayerNorm/BatchNorm weight & bias, and any other
        # bare vector parameter (e.g. our span-mask learnable token). None of
        # these should be weight-decayed -- standard practice, and especially
        # important here since our mask_token and positional-adjacent params
        # are exactly the kind of thing large uniform decay would damage.
        if param.ndim <= 1:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ], decay_params, no_decay_params
