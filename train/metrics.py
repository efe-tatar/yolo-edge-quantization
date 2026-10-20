import torch
from torchvision.ops import box_iou


def compute_ap(recall, precision):
    """Average Precision from a recall/precision curve, via the precision-envelope
    method (same as VOC2012/Ultralytics' reference implementation): make precision
    monotonically non-increasing from the right, then integrate over recall."""
    mrec = torch.cat([torch.tensor([0.0]), recall, torch.tensor([1.0])])
    mpre = torch.cat([torch.tensor([1.0]), precision, torch.tensor([0.0])])

    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = torch.maximum(mpre[i], mpre[i + 1])

    idx = torch.where(mrec[1:] != mrec[:-1])[0]
    ap = torch.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
    return ap.item()


def average_precision_per_class(preds, gts, class_id, iou_threshold):
    """preds: list of (boxes, scores, labels) per image (decode_predictions' output).
    gts: list of (boxes, labels) per image.

    Returns the AP for one class at one IoU threshold, or None if this class has
    no ground-truth instances anywhere (skipped from the mean, not scored as 0)."""
    gt_boxes_per_image = []
    gt_matched_per_image = []
    n_gt = 0

    for boxes, labels in gts:
        mask = labels == class_id
        gt_boxes_per_image.append(boxes[mask])
        gt_matched_per_image.append(torch.zeros(int(mask.sum()), dtype=torch.bool))
        n_gt += int(mask.sum())

    if n_gt == 0:
        return None

    candidates = []  # (score, img_idx, box)
    for img_idx, (boxes, scores, labels) in enumerate(preds):
        mask = labels == class_id
        for box, score in zip(boxes[mask], scores[mask]):
            candidates.append((score.item(), img_idx, box))

    if len(candidates) == 0:
        return 0.0  # GT instances exist but nothing was ever predicted -- true AP is 0, not 1

    candidates.sort(key=lambda c: c[0], reverse=True)

    tp = torch.zeros(len(candidates))
    fp = torch.zeros(len(candidates))

    for i, (_, img_idx, box) in enumerate(candidates):
        gt_boxes = gt_boxes_per_image[img_idx]
        if gt_boxes.shape[0] == 0:
            fp[i] = 1
            continue

        ious = box_iou(box.unsqueeze(0), gt_boxes)[0]
        best_iou, best_idx = ious.max(dim=0)

        if best_iou >= iou_threshold and not gt_matched_per_image[img_idx][best_idx]:
            tp[i] = 1
            gt_matched_per_image[img_idx][best_idx] = True
        else:
            fp[i] = 1

    tp_cumsum = torch.cumsum(tp, dim=0)
    fp_cumsum = torch.cumsum(fp, dim=0)

    recall = tp_cumsum / n_gt
    precision = tp_cumsum / torch.clamp(tp_cumsum + fp_cumsum, min=1e-9)

    return compute_ap(recall, precision)


def mean_average_precision(preds, gts, num_classes, iou_thresholds=(0.5,)):
    """Returns {iou_threshold: mAP, ...}, plus "mAP@[.5:.95]" when more than one
    threshold is given (COCO's primary metric: mean over thresholds 0.5:0.05:0.95)."""
    results = {}

    for iou_threshold in iou_thresholds:
        aps = []
        for class_id in range(num_classes):
            ap = average_precision_per_class(preds, gts, class_id, iou_threshold)
            if ap is not None:
                aps.append(ap)
        results[iou_threshold] = sum(aps) / len(aps) if aps else 0.0

    if len(iou_thresholds) > 1:
        results["mAP@[.5:.95]"] = sum(results[t] for t in iou_thresholds) / len(iou_thresholds)

    return results
