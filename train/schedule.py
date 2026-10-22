import math


def warmup_cosine(base_lr, epochs, warmup_epochs=3.0, final_lr_ratio=0.01):
    """Returns lr_at(t): t is training progress in epochs, fractional (epoch 2, batch
    50 of 100 -> t = 2.5), so warmup moves every batch, not every epoch.

    Linear warmup from ~0 to base_lr over warmup_epochs, then cosine decay down to
    base_lr * final_lr_ratio at t = epochs. A pure function of t on purpose: nothing
    to save in a checkpoint, a resumed run just recomputes it from the epoch number.
    """
    final_lr = base_lr * final_lr_ratio

    def lr_at(t):
        if warmup_epochs > 0 and t < warmup_epochs:
            return base_lr * t / warmup_epochs
        span = max(epochs - warmup_epochs, 1e-9)
        progress = min(max((t - warmup_epochs) / span, 0.0), 1.0)
        return final_lr + (base_lr - final_lr) * 0.5 * (1 + math.cos(math.pi * progress))

    return lr_at


def param_groups(model, weight_decay):
    """Weight decay on conv/linear weights only. Biases and BatchNorm parameters are
    excluded, the standard practice: decaying a BN scale toward 0 or a bias toward 0
    doesn't regularize anything, it just fights the normalization."""
    decay, no_decay = [], []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        (decay if param.ndim > 1 else no_decay).append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
