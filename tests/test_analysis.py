import torch
from PIL import Image

from train.analysis import (
    best_f1, confusion_pairs, manifest_gts, match_all, per_class_ap, precision_recall_f1,
    predict_own_model, recall_by_size,
)


def pred(boxes, scores, labels):
    return (torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            torch.tensor(scores, dtype=torch.float32), torch.tensor(labels, dtype=torch.long))


def gt(boxes, labels):
    return (torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4), torch.tensor(labels, dtype=torch.long))


A = [0, 0, 10, 10]
B = [20, 20, 30, 30]
NEAR_A = [1, 1, 11, 11]  # IoU with A = 81/119 ~ 0.68


def test_match_all_counts_duplicates_and_wrong_classes_as_false_positives():
    preds = [pred([A, NEAR_A, B], [0.9, 0.8, 0.7], [0, 0, 1])]
    gts = [gt([A, B], [0, 2])]  # B's true class is 2, predicted 1

    scores, is_tp, n_gt = match_all(preds, gts)
    assert n_gt == 2
    assert is_tp.tolist() == [True, False, False]  # duplicate of A and wrong-class B are FPs

    _, is_tp_agnostic, _ = match_all(preds, gts, class_aware=False)
    assert is_tp_agnostic.tolist() == [True, False, True]


def test_single_pass_matching_equals_matching_at_each_threshold():
    # a high-scoring wrong box and a lower-scoring right one competing for the same GT
    preds = [pred([NEAR_A, A, B], [0.9, 0.6, 0.3], [0, 0, 1]), pred([B], [0.5], [1])]
    gts = [gt([A, B], [0, 1]), gt([B], [1])]

    scores, is_tp, n_gt = match_all(preds, gts, min_conf=0.0)
    for t in (0.2, 0.4, 0.55, 0.7, 0.95):
        direct_scores, direct_tp, _ = match_all(preds, gts, min_conf=t)
        from_single = precision_recall_f1(scores, is_tp, n_gt, t)
        direct = precision_recall_f1(direct_scores, direct_tp, n_gt, t)
        assert from_single == direct


def test_best_f1_is_perfect_on_perfect_predictions():
    preds = [pred([A, B], [0.9, 0.8], [0, 1])]
    gts = [gt([A, B], [0, 1])]

    best, rows = best_f1(preds, gts)
    assert best["f1"] == 1.0 and best["precision"] == 1.0 and best["recall"] == 1.0
    assert len(rows) == 22


def test_best_f1_picks_the_threshold_that_drops_low_confidence_junk():
    preds = [pred([A, B, [50, 50, 60, 60]], [0.9, 0.8, 0.2], [0, 1, 0])]
    gts = [gt([A, B], [0, 1])]

    best, _ = best_f1(preds, gts)
    assert best["f1"] == 1.0 and 0.2 < best["conf"] <= 0.8


def test_recall_by_size_bins():
    small, big = [0, 0, 5, 5], [0, 0, 50, 50]
    preds = [pred([big], [0.9], [0])]          # finds the big one only
    gts = [gt([small, big], [0, 0])]
    sizes = [[5.0, 50.0]]

    rows = recall_by_size(preds, gts, sizes, edges=(16, 32), conf_thresh=0.5)
    assert rows == [("0-16", 1, 0.0), ("16-32", 0, rows[1][2]), (">=32", 1, 1.0)]
    assert rows[1][2] != rows[1][2]  # empty bin -> nan


def test_confusion_pairs_counts_right_box_wrong_class():
    preds = [pred([A, B], [0.9, 0.9], [3, 1]), pred([A], [0.9], [3])]
    gts = [gt([A, B], [0, 1]), gt([A], [0])]

    assert confusion_pairs(preds, gts, conf_thresh=0.5) == {(0, 3): 2}


def test_per_class_ap_skips_absent_classes():
    preds = [pred([A], [0.9], [0])]
    gts = [gt([A], [0])]
    assert per_class_ap(preds, gts, num_classes=2) == [1.0, None]


class TargetEcho(torch.nn.Module):
    """Outputs, as if predicted, the training targets of a fixed batch: decoding them
    must recover the GT boxes exactly."""

    def __init__(self, targets):
        super().__init__()
        self.targets = targets

    def forward(self, images):
        return [(box_t, torch.where(obj_t > 0, 8.0, -8.0), cls_t * 16.0 - 8.0)
                for box_t, obj_t, cls_t in self.targets]


def test_predict_own_model_maps_boxes_back_to_original_pixels(tmp_path):
    from data.dataset import DetectionDataset, collate_fn
    from train.targets import build_targets_batch

    # a non-square image: x and y are rescaled differently on the way to 64x64
    path = tmp_path / "img.png"
    Image.new("RGB", (200, 100)).save(path)
    samples = [{"image_path": str(path), "boxes": [[40.0, 20.0, 120.0, 70.0]], "labels": [1]}]

    images, boxes, labels = collate_fn([DetectionDataset(samples, img_size=64)[0]])
    targets = build_targets_batch(boxes, labels, 64, 64, (8, 16, 32), 3, (16, 32))

    preds = predict_own_model(TargetEcho(targets), samples, img_size=64, device="cpu", conf_thresh=0.5)
    (out_boxes, out_scores, out_labels), = preds

    assert out_labels.tolist() == [1]
    assert torch.allclose(out_boxes, manifest_gts(samples)[0][0], atol=1e-3)
