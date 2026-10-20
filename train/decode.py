import torchvision
import torch
from train.targets import decode_boxes

def decode_predictions(preds, strides, conf_thresh=0.3, iou_thresh=0.5):
    batch_size = preds[0][0].shape[0]
    results = []

    for b in range(batch_size):
        level_boxes, level_scores, level_labels = [], [], []

        for (box_pred, cls_pred), stride in zip(preds, strides):
            decoded = decode_boxes(box_pred[b:b+1], stride)        # (1, H, W, 4)
            conf, cls_idx = torch.sigmoid(cls_pred[b]).max(dim=0)  # (H, W) each

            boxes = decoded.reshape(-1, 4)
            scores = conf.reshape(-1)
            labels = cls_idx.reshape(-1)

            keep = scores > conf_thresh
            level_boxes.append(boxes[keep])
            level_scores.append(scores[keep])
            level_labels.append(labels[keep])

        all_boxes = torch.cat(level_boxes, dim=0)
        all_scores = torch.cat(level_scores, dim=0)
        all_labels = torch.cat(level_labels, dim=0)

        if all_boxes.shape[0] == 0:
            results.append((all_boxes, all_scores, all_labels))
            continue

        keep_idx = torchvision.ops.batched_nms(all_boxes, all_scores, all_labels, iou_thresh)
        results.append((all_boxes[keep_idx], all_scores[keep_idx], all_labels[keep_idx]))

    return results


