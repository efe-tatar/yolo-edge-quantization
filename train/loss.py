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


def yolo_loss(preds, targets, box_weight=5.0, obj_weight=1.0, noobj_weight=0.5, cls_norm="positive"):
    """Multi-scale, multi-class detection loss -- decoupled objectness (YOLOv1-style),
    not the merged cls-is-also-objectness design we started with. That merged design
    needed focal_loss as a patch because a 54-channel classifier was getting a BCE
    term at every one of ~2,100 cells per image, ~99.98% of them negative. Decoupling
    removes classification from facing that imbalance at all: it only ever sees
    positive cells now. Objectness still faces it, but as a single channel (not 54),
    plain weighted BCE (noobj_weight < obj_weight) is the classic, long-proven fix
    there (YOLOv1 through v5 all trained this way, well before focal loss existed).

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

    total = 0.0
    obj_loss_sum = 0.0
    cls_loss_sum = 0.0
    box_loss_sum = 0.0

    for (box_pred, obj_pred, cls_pred), (box_target, obj_target, cls_target) in zip(preds, targets):
        obj_bce = F.binary_cross_entropy_with_logits(obj_pred, obj_target, reduction="none")
        weight = torch.where(obj_target > 0, obj_weight, noobj_weight)
        obj_loss = (obj_bce * weight).mean()

        pos_mask = obj_target > 0  # (B,1,H,W) -- ground truth, not the model's own guess

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
