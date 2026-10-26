"""Model-agnostic analysis for the comparison notebook. Everything here works on the
same plain format, whichever model produced it:

    preds: list per image of (boxes (N, 4) xyxy, scores (N,), labels (N,))
    gts:   list per image of (boxes (M, 4) xyxy, labels (M,))

in ORIGINAL image pixels -- the frame Ultralytics returns, and the manifest's. This
repo's own model works in a 320x320 stretched frame; predict_own_model maps its boxes
back. IoU doesn't change when x and y are each rescaled, so neither model is favoured.
"""
import torch
from PIL import Image
from torchvision.ops import box_iou

from data.dataset import DetectionDataset, collate_fn
from train.decode import decode_predictions
from train.metrics import average_precision_per_class

OWN_MODEL_STRIDES = (8, 16, 32)  # models/yolo.py: heads on P3/P4/P5


def manifest_gts(samples):
    """Ground truth straight from manifest samples, in original image pixels."""
    return [
        (torch.tensor(s["boxes"], dtype=torch.float32).reshape(-1, 4),
         torch.tensor(s["labels"], dtype=torch.long))
        for s in samples
    ]


def image_sizes(samples):
    """(width, height) of each sample's original image -- header read only."""
    sizes = []
    for s in samples:
        with Image.open(s["image_path"]) as image:
            sizes.append(image.size)
    return sizes


@torch.no_grad()
def predict_own_model(model, samples, img_size, device, conf_thresh=0.001, nms_iou=0.5,
                      batch_size=16):
    """Runs this repo's model on samples (no augmentation) and returns preds with boxes
    mapped back from the img_size x img_size stretched frame to original pixels."""
    model.eval()
    sizes = image_sizes(samples)
    loader = torch.utils.data.DataLoader(
        DetectionDataset(samples, img_size=img_size), batch_size=batch_size,
        shuffle=False, collate_fn=collate_fn,
    )

    preds = []
    for images, _, _ in loader:
        decoded = decode_predictions(model(images.to(device)), OWN_MODEL_STRIDES,
                                     conf_thresh=conf_thresh, iou_thresh=nms_iou)
        for boxes, scores, labels in decoded:
            w, h = sizes[len(preds)]
            boxes = boxes.cpu().clone()
            boxes[:, [0, 2]] *= w / img_size
            boxes[:, [1, 3]] *= h / img_size
            preds.append((boxes, scores.cpu(), labels.cpu()))
    return preds


def per_class_ap(preds, gts, num_classes, iou_threshold=0.5):
    """AP per class (None where a class has no GT in this split)."""
    return [average_precision_per_class(preds, gts, c, iou_threshold) for c in range(num_classes)]


def _greedy_match(boxes, scores, labels, gt_boxes, gt_labels, iou_threshold, class_aware):
    """Standard detection matching: predictions in descending score order, each takes
    the best still-unmatched GT with IoU >= iou_threshold (same class if class_aware).
    Returns pred_to_gt: for each prediction, the GT index it matched, or -1."""
    pred_to_gt = torch.full((len(scores),), -1, dtype=torch.long)
    if len(scores) == 0 or len(gt_boxes) == 0:
        return pred_to_gt

    ious = box_iou(boxes, gt_boxes)
    if class_aware:
        ious = ious * (labels[:, None] == gt_labels[None, :])
    gt_taken = torch.zeros(len(gt_boxes), dtype=torch.bool)

    for i in scores.argsort(descending=True).tolist():
        candidates = ious[i].masked_fill(gt_taken, -1.0)
        best = int(candidates.argmax())
        if candidates[best] >= iou_threshold:
            pred_to_gt[i] = best
            gt_taken[best] = True
    return pred_to_gt


def match_all(preds, gts, min_conf=0.0, iou_threshold=0.5, class_aware=True):
    """Greedy-matches every prediction above min_conf once. Returns (scores, is_tp,
    n_gt) over all images.

    Because matching goes in descending score order, the matches made by predictions
    above any threshold t >= min_conf are exactly the ones made in this single pass --
    so precision/recall at every threshold come from one matching, not one per t.
    """
    all_scores, all_tp, n_gt = [], [], 0
    for (boxes, scores, labels), (gt_boxes, gt_labels) in zip(preds, gts):
        keep = scores >= min_conf
        pred_to_gt = _greedy_match(boxes[keep], scores[keep], labels[keep], gt_boxes, gt_labels,
                                   iou_threshold, class_aware)
        all_scores.append(scores[keep])
        all_tp.append(pred_to_gt >= 0)
        n_gt += len(gt_boxes)
    return torch.cat(all_scores), torch.cat(all_tp), n_gt


def precision_recall_f1(scores, is_tp, n_gt, conf_thresh):
    above = scores >= conf_thresh
    tp = int((is_tp & above).sum())
    fp = int(above.sum()) - tp
    precision = tp / max(tp + fp, 1)
    recall = tp / max(n_gt, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return {"conf": conf_thresh, "precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "fp": fp, "fn": n_gt - tp}


def best_f1(preds, gts, thresholds=None, iou_threshold=0.5, class_aware=True):
    """The confidence threshold maximizing F1 (micro, over all boxes) -- a model's
    natural deployment operating point. Returns (best row, rows for every threshold)."""
    # finer near the top: a confident model's optimum can sit above 0.95
    thresholds = thresholds or [round(0.05 * i, 2) for i in range(1, 20)] + [0.97, 0.98, 0.99]
    scores, is_tp, n_gt = match_all(preds, gts, min(thresholds), iou_threshold, class_aware)
    rows = [precision_recall_f1(scores, is_tp, n_gt, t) for t in thresholds]
    return max(rows, key=lambda r: r["f1"]), rows


def recall_by_size(preds, gts, gt_sizes, edges, conf_thresh, iou_threshold=0.5, class_aware=True):
    """Recall per GT size bin. gt_sizes: list per image of (M,) sizes, in whatever unit
    edges use (e.g. max side in model-input pixels). edges (e0, e1, ...) define bins
    [0, e0), [e0, e1), ..., [e_last, inf). Returns [(label, n_gt, recall), ...]."""
    bounds = [0.0, *edges, float("inf")]
    matched, sizes = [], []
    for (boxes, scores, labels), (gt_boxes, gt_labels), gt_size in zip(preds, gts, gt_sizes):
        keep = scores >= conf_thresh
        pred_to_gt = _greedy_match(boxes[keep], scores[keep], labels[keep], gt_boxes, gt_labels,
                                   iou_threshold, class_aware)
        hit = torch.zeros(len(gt_boxes), dtype=torch.bool)
        hit[pred_to_gt[pred_to_gt >= 0]] = True
        matched.append(hit)
        sizes.append(torch.as_tensor(gt_size, dtype=torch.float32))
    matched, sizes = torch.cat(matched), torch.cat(sizes)

    rows = []
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        in_bin = (sizes >= lo) & (sizes < hi)
        label = f"{lo:g}-{hi:g}" if hi != float("inf") else f">={lo:g}"
        n = int(in_bin.sum())
        rows.append((label, n, float(matched[in_bin].float().mean()) if n else float("nan")))
    return rows


def confusion_pairs(preds, gts, conf_thresh, iou_threshold=0.5):
    """Counts (gt_class, predicted_class) for predictions that found an object (class-
    agnostic match, IoU >= iou_threshold) but gave it the wrong class. Returns a dict
    {(gt, pred): count}, i.e. what the model mistakes each class for."""
    counts = {}
    for (boxes, scores, labels), (gt_boxes, gt_labels) in zip(preds, gts):
        keep = scores >= conf_thresh
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
        pred_to_gt = _greedy_match(boxes, scores, labels, gt_boxes, gt_labels,
                                   iou_threshold, class_aware=False)
        for i, g in enumerate(pred_to_gt.tolist()):
            if g >= 0 and int(labels[i]) != int(gt_labels[g]):
                key = (int(gt_labels[g]), int(labels[i]))
                counts[key] = counts.get(key, 0) + 1
    return counts
