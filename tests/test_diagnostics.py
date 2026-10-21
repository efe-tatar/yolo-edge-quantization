import math

import torch

from models.yolo import YOLO
from train.diagnostics import branch_stats

IMG_SIZE = 64
STRIDES = (8, 16, 32)
THRESHOLDS = (16, 32)
NUM_CLASSES = 3


def make_batch():
    images = torch.rand(2, 3, IMG_SIZE, IMG_SIZE)
    # sizes 15 and 5 -> level 0, size 20 -> level 1, nothing on level 2
    boxes_list = [
        torch.tensor([[5.0, 5.0, 20.0, 20.0]]),
        torch.tensor([[10.0, 10.0, 15.0, 15.0], [30.0, 30.0, 50.0, 50.0]]),
    ]
    labels_list = [torch.tensor([0]), torch.tensor([1, 2])]
    return images, boxes_list, labels_list


class FixedHead(torch.nn.Module):
    """Ignores its input and returns hand-set logits, so the expected stats are known
    exactly: obj is high only at the true positive cells, cls always says class 1."""

    def __init__(self, targets):
        super().__init__()
        self.targets = targets

    def forward(self, images):
        preds = []
        for box_t, obj_t, cls_t in self.targets:
            obj_logit = torch.where(obj_t > 0, 4.0, -4.0)
            cls_logit = torch.full_like(cls_t, -4.0)
            cls_logit[:, 1] = 4.0
            preds.append((box_t.clone(), obj_logit, cls_logit))
        return preds


def test_branch_stats_counts_positives_per_level():
    results = branch_stats(YOLO(NUM_CLASSES), [make_batch()], "cpu", IMG_SIZE, STRIDES,
                           NUM_CLASSES, THRESHOLDS)

    assert results["level0"]["n_pos"] == 2
    assert results["level1"]["n_pos"] == 1
    assert results["level2"]["n_pos"] == 0
    assert results["all"]["n_pos"] == 3
    assert math.isnan(results["level2"]["obj_pos_median"])  # no positives -> nan, not a crash


def test_branch_stats_reads_the_right_cells():
    from train.targets import build_targets_batch

    images, boxes_list, labels_list = make_batch()
    targets = build_targets_batch(boxes_list, labels_list, IMG_SIZE, IMG_SIZE, STRIDES,
                                  NUM_CLASSES, THRESHOLDS)
    model = FixedHead(targets)

    stats = branch_stats(model, [(images, boxes_list, labels_list)], "cpu", IMG_SIZE, STRIDES,
                         NUM_CLASSES, THRESHOLDS)["all"]

    assert stats["obj_pos_median"] > 0.98  # sigmoid(4)
    assert stats["obj_neg_mean"] < 0.02  # sigmoid(-4)
    assert stats["neg_above_pos_median"] == 0.0
    assert abs(stats["cls_top1_acc"] - 1 / 3) < 1e-6  # always predicts class 1; one of 3 GT is class 1

    # GT labels are 0, 1, 2 once each -> any majority is a third
    assert abs(stats["majority_share"] - 1 / 3) < 1e-6
    # the classifier collapsed onto class 1
    assert stats["pred_mode_class"] == 1
    assert stats["pred_mode_share"] == 1.0
    assert stats["n_distinct_pred"] == 1

    # FixedHead predicts the target boxes exactly
    assert abs(stats["box_iou_mean"] - 1.0) < 1e-5
    assert stats["box_iou>=0.5"] == 1.0


def test_paired_iou_is_elementwise():
    from train.diagnostics import paired_iou

    a = torch.tensor([[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 10.0, 10.0]])
    b = torch.tensor([[0.0, 0.0, 10.0, 10.0], [5.0, 0.0, 15.0, 10.0]])

    iou = paired_iou(a, b)
    assert iou.shape == (2,)
    assert abs(iou[0].item() - 1.0) < 1e-6
    assert abs(iou[1].item() - 50 / 150) < 1e-6  # 5x10 overlap over 100+100-50 union
