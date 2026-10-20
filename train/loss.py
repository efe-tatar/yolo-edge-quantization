import torch
import torch.nn.functional as F


def yolo_loss(preds, targets, box_weight=5.0):
    """Multi-scale, multi-class detection loss.

    preds, targets: each a list of (box, cls) pairs, one per level (same order as
    strides/levels used to build the targets).
        box: (B, 4, H, W) -- raw (l, t, r, b) in units of that level's stride.
        cls: (B, num_classes, H, W) -- raw logits (preds) / one-hot-or-zero (targets).

    A cell is positive iff any class channel is set in its target -- there's no
    separate objectness branch, "is there an object" is implicit in the class scores.
    """
    total = 0.0
    cls_loss_sum = 0.0
    box_loss_sum = 0.0

    for (box_pred, cls_pred), (box_target, cls_target) in zip(preds, targets):
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
