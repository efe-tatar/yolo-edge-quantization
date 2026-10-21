import torch

from train.targets import build_targets_batch, decode_boxes


def paired_iou(a, b):
    """Elementwise IoU of two (N, 4) xyxy box sets -- box i against box i only, unlike
    torchvision's box_iou which is all-pairs (N, M)."""
    x1 = torch.maximum(a[:, 0], b[:, 0])
    y1 = torch.maximum(a[:, 1], b[:, 1])
    x2 = torch.minimum(a[:, 2], b[:, 2])
    y2 = torch.minimum(a[:, 3], b[:, 3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a + area_b - inter).clamp(min=1e-9)


def _summarize(obj_pos, obj_neg, cls_true, cls_max, true_idx, pred_idx, iou, num_classes):
    """All inputs are 1-D tensors over cells (obj_neg over negative cells, the rest
    over positive ones). Medians of an empty tensor are undefined -> nan, not a crash,
    since a level can legitimately have no positives in a small val set."""
    def median(t):
        return t.median().item() if t.numel() else float("nan")

    def mean(t):
        return t.mean().item() if t.numel() else float("nan")

    def mode(t):
        """(most frequent value, its share of t)."""
        if not t.numel():
            return -1, float("nan")
        counts = torch.bincount(t, minlength=num_classes)
        return counts.argmax().item(), (counts.max() / t.numel()).item()

    pos_median = median(obj_pos)
    majority_class, majority_share = mode(true_idx)
    pred_mode_class, pred_mode_share = mode(pred_idx)

    return {
        "n_pos": obj_pos.numel(),
        "obj_pos_mean": mean(obj_pos),
        "obj_pos_median": pos_median,
        "obj_neg_mean": mean(obj_neg),
        # ranking, not calibration: what fraction of empty cells the model rates as
        # more object-like than a typical real object -- ~0 is good, ~0.5 is a coin flip
        "neg_above_pos_median": mean((obj_neg > pos_median).float()),
        "cls_true_median": median(cls_true),
        "cls_max_median": median(cls_max),
        "cls_top1_acc": mean((pred_idx == true_idx).float()),
        # the "always guess the most common class" baseline cls_top1_acc must beat
        "majority_class": majority_class,
        "majority_share": majority_share,
        # if this is near 1, the classifier has collapsed onto one answer
        "pred_mode_class": pred_mode_class,
        "pred_mode_share": pred_mode_share,
        "n_distinct_pred": int(pred_idx.unique().numel()),
        # predicted box at the assigned cell vs. its GT box -- localization quality,
        # independent of objectness/class
        "box_iou_mean": mean(iou),
        "box_iou_median": median(iou),
        "box_iou>=0.5": mean((iou >= 0.5).float()),
        # what decode_predictions would score a true positive at, before thresholding
        "score_pos_median": median(obj_pos * cls_max),
    }


@torch.no_grad()
def branch_stats(model, dataloader, device, img_size, strides, num_classes, thresholds):
    """Reads each head branch at the ground-truth cells, bypassing decode/NMS/mAP
    entirely -- so a 0.0 mAP can be traced to the branch actually responsible:

        obj: sigmoid(obj) at positive vs. negative cells (is objectness separating them?)
        cls: top-1 accuracy and confidence at positive cells (is the class right, given
             we already know where the object is?), against a majority-class baseline
        box: IoU of the predicted box at each positive cell with its GT box

    Returns {"level0": {...}, "level1": {...}, ..., "all": {...}}, see _summarize.
    """
    model.eval()
    keys = ["obj_pos", "obj_neg", "cls_true", "cls_max", "true_idx", "pred_idx", "iou"]
    acc = [{k: [] for k in keys} for _ in strides]

    for images, boxes_list, labels_list in dataloader:
        preds = model(images.to(device))
        targets = build_targets_batch(
            boxes_list, labels_list, img_size, img_size, strides, num_classes, thresholds
        )

        for level, ((box_pred, obj_pred, cls_pred), (box_target, obj_target, cls_target), stride) in enumerate(
            zip(preds, targets, strides)
        ):
            obj_prob = torch.sigmoid(obj_pred[:, 0]).cpu()               # (B, H, W)
            cls_prob = torch.sigmoid(cls_pred).cpu().permute(0, 2, 3, 1)  # (B, H, W, C)
            pos = obj_target[:, 0] > 0                                  # (B, H, W)

            cls_prob_pos = cls_prob[pos]                                 # (P, C)
            true_idx = cls_target.permute(0, 2, 3, 1)[pos].argmax(dim=1)  # (P,)
            cls_max, pred_idx = cls_prob_pos.max(dim=1)

            # box targets share the prediction's (l, t, r, b) parameterization, so the
            # same decode turns both into absolute boxes
            pred_boxes = decode_boxes(box_pred.cpu(), stride)[pos]      # (P, 4)
            gt_boxes = decode_boxes(box_target, stride)[pos]            # (P, 4)

            acc[level]["obj_pos"].append(obj_prob[pos])
            acc[level]["obj_neg"].append(obj_prob[~pos])
            acc[level]["cls_true"].append(cls_prob_pos.gather(1, true_idx.unsqueeze(1)).squeeze(1))
            acc[level]["cls_max"].append(cls_max)
            acc[level]["true_idx"].append(true_idx)
            acc[level]["pred_idx"].append(pred_idx)
            acc[level]["iou"].append(paired_iou(pred_boxes, gt_boxes))

    per_level = [{k: torch.cat(v) for k, v in level_acc.items()} for level_acc in acc]

    results = {f"level{i}": _summarize(**lv, num_classes=num_classes) for i, lv in enumerate(per_level)}
    results["all"] = _summarize(
        **{k: torch.cat([lv[k] for lv in per_level]) for k in keys}, num_classes=num_classes
    )
    return results
