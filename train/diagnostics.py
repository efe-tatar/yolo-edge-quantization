import torch

from train.targets import build_targets_batch


def _summarize(obj_pos, obj_neg, cls_true, cls_max, top1):
    """All inputs are 1-D tensors over cells (obj_neg over negative cells, the rest
    over positive ones). Medians of an empty tensor are undefined -> nan, not a crash,
    since a level can legitimately have no positives in a small val set."""
    def median(t):
        return t.median().item() if t.numel() else float("nan")

    def mean(t):
        return t.mean().item() if t.numel() else float("nan")

    pos_median = median(obj_pos)
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
        "cls_top1_acc": mean(top1.float()),
        # what decode_predictions would score a true positive at, before thresholding
        "score_pos_median": median(obj_pos * cls_max),
    }


@torch.no_grad()
def branch_stats(model, dataloader, device, img_size, strides, num_classes, thresholds):
    """Reads each head branch at the ground-truth cells, bypassing decode/NMS/mAP
    entirely -- so a 0.0 mAP can be traced to the branch actually responsible:

        obj: sigmoid(obj) at positive vs. negative cells (is objectness separating them?)
        cls: top-1 accuracy and confidence at positive cells (is the class right, given
             we already know where the object is?)

    Returns {"level0": {...}, "level1": {...}, ..., "all": {...}}, see _summarize.
    """
    model.eval()
    n_levels = len(strides)
    acc = [{"obj_pos": [], "obj_neg": [], "cls_true": [], "cls_max": [], "top1": []}
           for _ in range(n_levels)]

    for images, boxes_list, labels_list in dataloader:
        preds = model(images.to(device))
        targets = build_targets_batch(
            boxes_list, labels_list, img_size, img_size, strides, num_classes, thresholds
        )

        for level, ((_, obj_pred, cls_pred), (_, obj_target, cls_target)) in enumerate(zip(preds, targets)):
            obj_prob = torch.sigmoid(obj_pred[:, 0]).cpu()               # (B, H, W)
            cls_prob = torch.sigmoid(cls_pred).cpu().permute(0, 2, 3, 1)  # (B, H, W, C)
            pos = obj_target[:, 0] > 0                                  # (B, H, W)

            cls_prob_pos = cls_prob[pos]                                 # (P, C)
            true_idx = cls_target.permute(0, 2, 3, 1)[pos].argmax(dim=1)  # (P,)
            cls_max, cls_argmax = cls_prob_pos.max(dim=1)

            acc[level]["obj_pos"].append(obj_prob[pos])
            acc[level]["obj_neg"].append(obj_prob[~pos])
            acc[level]["cls_true"].append(cls_prob_pos.gather(1, true_idx.unsqueeze(1)).squeeze(1))
            acc[level]["cls_max"].append(cls_max)
            acc[level]["top1"].append(cls_argmax == true_idx)

    per_level = [{k: torch.cat(v) for k, v in level_acc.items()} for level_acc in acc]

    results = {f"level{i}": _summarize(**lv) for i, lv in enumerate(per_level)}
    results["all"] = _summarize(**{k: torch.cat([lv[k] for lv in per_level]) for k in per_level[0]})
    return results
