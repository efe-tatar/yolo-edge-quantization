import torch
import torch.nn.functional as F


def focal_loss(logits, targets, alpha=0.25, gamma=2.0):
    """RetinaNet's Focal Loss: down-weights the loss contribution of easy, confident
    examples (overwhelmingly negatives, in a dense per-cell head like this one) so
    the rare positives aren't drowned out. Plain BCE can "converge" by learning to
    predict near-zero everywhere, since ~2,100 cells x num_classes channels per image
    are negative and only a handful are positive -- the average loss looks fine while
    the model never actually learns to detect anything.

    alpha=0.25, gamma=2.0 are RetinaNet's published defaults, not tuned for this
    dataset -- reasonable starting point, not a claim they're optimal here.
    """
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p = torch.sigmoid(logits)
    p_t = p * targets + (1 - p) * (1 - targets)  # the model's confidence in the TRUE label
    modulating = (1 - p_t) ** gamma  # near 0 for confident-correct, near 1 for wrong/unsure
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    return (alpha_t * modulating * bce).mean()


def yolo_loss(preds, targets, box_weight=5.0, use_focal=True, alpha=0.25, gamma=2.0):
    """Multi-scale, multi-class detection loss.

    preds, targets: each a list of (box, cls) pairs, one per level (same order as
    strides/levels used to build the targets).
        box: (B, 4, H, W) -- raw (l, t, r, b) in units of that level's stride.
        cls: (B, num_classes, H, W) -- raw logits (preds) / one-hot-or-zero (targets).

    A cell is positive iff any class channel is set in its target -- there's no
    separate objectness branch, "is there an object" is implicit in the class scores.

    use_focal=False falls back to plain BCE -- kept as a toggle, not removed, so a
    focal-vs-plain-BCE ablation is one flag away rather than a code change.
    """
    total = 0.0
    cls_loss_sum = 0.0
    box_loss_sum = 0.0

    for (box_pred, cls_pred), (box_target, cls_target) in zip(preds, targets):
        if use_focal:
            cls_loss = focal_loss(cls_pred, cls_target, alpha=alpha, gamma=gamma)
        else:
            cls_loss = F.binary_cross_entropy_with_logits(cls_pred, cls_target)

        pos_mask = cls_target.sum(dim=1, keepdim=True) > 0  # (B,1,H,W)
        pos_mask_box = pos_mask.expand_as(box_pred)
        if pos_mask_box.any():
            box_loss = F.l1_loss(box_pred[pos_mask_box], box_target[pos_mask_box])
        else:
            box_loss = box_pred.sum() * 0.0  # keeps it a tensor on the right device/graph

        total = total + cls_loss + box_weight * box_loss
        cls_loss_sum = cls_loss_sum + cls_loss.detach()
        box_loss_sum = box_loss_sum + box_loss.detach()

    return {"total": total, "cls": cls_loss_sum, "box": box_loss_sum}
