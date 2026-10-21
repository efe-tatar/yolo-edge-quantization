import torch
import torch.nn.functional as F


def focal_loss(logits, targets, alpha=0.25, gamma=2.0):
    """RetinaNet's Focal Loss -- kept as a standalone, reusable, tested utility even
    though yolo_loss no longer uses it by default (see module docstring below for why).
    Down-weights confident-correct examples so rare/hard ones aren't drowned out.
    """
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p = torch.sigmoid(logits)
    p_t = p * targets + (1 - p) * (1 - targets)
    modulating = (1 - p_t) ** gamma
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    return (alpha_t * modulating * bce).mean()


def yolo_loss(preds, targets, box_weight=5.0, obj_weight=1.0, noobj_weight=0.5, cls_norm="positive",
              obj_norm="balanced"):
    """Multi-scale, multi-class detection loss -- decoupled objectness (YOLOv1-style),
    not the merged cls-is-also-objectness design we started with. That merged design
    needed focal_loss as a patch because a 54-channel classifier was getting a BCE
    term at every one of ~2,100 cells per image, ~99.98% of them negative. Decoupling
    removes classification from facing that imbalance at all: it only ever sees
    positive cells now. Objectness still faces it, as a single channel (not 54) --
    see obj_norm for how.

    obj_norm: how positive and negative cells are combined in the obj BCE.
        "balanced" (default) -- obj_weight * mean over positive cells
            + noobj_weight * mean over negative cells. Each side averaged on its own,
            so the positive term's pull doesn't depend on how many empty cells exist.
        "element" -- one weighted mean over all cells. Our original, YOLOv1's form,
            kept only for comparison. YOLOv1 had 49 cells per image; we have ~2,100
            with 1-2 positives, so positives are ~0.05% of that mean and a 0.5
            noobj_weight can't compensate -- measured on a 64-image overfit test:
            objectness at true object cells collapsed to a median of ~0.001, and mAP
            on the memorized images stalled at 0.25 despite 100% cls accuracy and
            0.87 box IoU.

    cls_norm: what the summed cls BCE is divided by.
        "positive" (default) -- the number of positive cells, YOLOv8's normalization.
            Each positive cell contributes its full BCE summed over all classes.
        "element" -- positive cells x num_classes, i.e. a plain .mean(). Our original
            choice, kept only for comparison: it shrinks the one correct class's
            gradient by 1/num_classes (1/54 here) while the num_classes-1 "not this
            class" targets keep pulling every logit down -- measured result after 26
            epochs: the classifier collapsed onto the most frequent class (59% of
            predictions for a class that's 12.5% of GT, top-1 ~19% on train AND val),
            while boxes (IoU median 0.82) and objectness trained fine.

    preds, targets: each a list of (box, obj, cls) triples, one per level (same
    order as strides/levels used to build the targets).
        box: (B, 4, H, W) -- raw (l, t, r, b) in units of that level's stride.
        obj: (B, 1, H, W) -- raw objectness logit (preds) / 0-or-1 (targets).
        cls: (B, num_classes, H, W) -- raw logits (preds) / one-hot-or-zero (targets).
    """
    if cls_norm not in ("positive", "element"):
        raise ValueError(f"cls_norm must be 'positive' or 'element', got {cls_norm!r}")
    if obj_norm not in ("balanced", "element"):
        raise ValueError(f"obj_norm must be 'balanced' or 'element', got {obj_norm!r}")

    total = 0.0
    obj_loss_sum = 0.0
    cls_loss_sum = 0.0
    box_loss_sum = 0.0

    for (box_pred, obj_pred, cls_pred), (box_target, obj_target, cls_target) in zip(preds, targets):
        pos_mask = obj_target > 0  # (B,1,H,W) -- ground truth, not the model's own guess

        obj_bce = F.binary_cross_entropy_with_logits(obj_pred, obj_target, reduction="none")
        if obj_norm == "balanced":
            # a level with no positives in this batch just has no positive term
            pos_term = obj_bce[pos_mask].mean() if pos_mask.any() else obj_bce.sum() * 0.0
            neg_term = obj_bce[~pos_mask].mean() if (~pos_mask).any() else obj_bce.sum() * 0.0
            obj_loss = obj_weight * pos_term + noobj_weight * neg_term
        else:
            weight = torch.where(pos_mask, obj_weight, noobj_weight)
            obj_loss = (obj_bce * weight).mean()

        pos_mask_cls = pos_mask.expand_as(cls_pred)
        if pos_mask_cls.any():
            cls_bce = F.binary_cross_entropy_with_logits(
                cls_pred[pos_mask_cls], cls_target[pos_mask_cls], reduction="sum"
            )
            n = pos_mask.sum() if cls_norm == "positive" else pos_mask_cls.sum()
            cls_loss = cls_bce / n
        else:
            cls_loss = cls_pred.sum() * 0.0

        pos_mask_box = pos_mask.expand_as(box_pred)
        if pos_mask_box.any():
            box_loss = F.l1_loss(box_pred[pos_mask_box], box_target[pos_mask_box])
        else:
            box_loss = box_pred.sum() * 0.0

        total = total + obj_loss + cls_loss + box_weight * box_loss
        obj_loss_sum = obj_loss_sum + obj_loss.detach()
        cls_loss_sum = cls_loss_sum + cls_loss.detach()
        box_loss_sum = box_loss_sum + box_loss.detach()

    return {"total": total, "obj": obj_loss_sum, "cls": cls_loss_sum, "box": box_loss_sum}
