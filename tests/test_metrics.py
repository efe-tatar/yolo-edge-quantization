import torch

from train.metrics import average_precision_per_class, mean_average_precision


def test_perfect_prediction_gives_ap_one():
    gts = [(torch.tensor([[10.0, 10.0, 50.0, 50.0]]), torch.tensor([0]))]
    preds = [(torch.tensor([[10.0, 10.0, 50.0, 50.0]]), torch.tensor([0.9]), torch.tensor([0]))]

    ap = average_precision_per_class(preds, gts, class_id=0, iou_threshold=0.5)
    assert ap == 1.0


def test_no_predictions_for_a_class_with_gt_gives_ap_zero():
    # the known trap: this must NOT fall through to compute_ap's general formula,
    # which would wrongly give 1.0 for an empty prediction list
    gts = [(torch.tensor([[10.0, 10.0, 50.0, 50.0]]), torch.tensor([0]))]
    preds = [(torch.zeros(0, 4), torch.zeros(0), torch.zeros(0, dtype=torch.long))]

    ap = average_precision_per_class(preds, gts, class_id=0, iou_threshold=0.5)
    assert ap == 0.0


def test_class_absent_from_ground_truth_returns_none():
    gts = [(torch.tensor([[10.0, 10.0, 50.0, 50.0]]), torch.tensor([0]))]
    preds = [(torch.tensor([[10.0, 10.0, 50.0, 50.0]]), torch.tensor([0.9]), torch.tensor([1]))]

    ap = average_precision_per_class(preds, gts, class_id=1, iou_threshold=0.5)
    assert ap is None


def test_completely_wrong_predictions_give_ap_zero():
    gts = [(torch.tensor([[10.0, 10.0, 50.0, 50.0]]), torch.tensor([0]))]
    preds = [(torch.tensor([[200.0, 200.0, 220.0, 220.0]]), torch.tensor([0.9]), torch.tensor([0]))]

    ap = average_precision_per_class(preds, gts, class_id=0, iou_threshold=0.5)
    assert ap == 0.0


def test_predictions_only_match_ground_truth_in_their_own_image():
    # image 0's prediction coincidentally has the same coordinates as image 1's GT box,
    # but must only be checked against image 0's own (non-overlapping) GT
    gts = [
        (torch.tensor([[0.0, 0.0, 10.0, 10.0]]), torch.tensor([0])),
        (torch.tensor([[100.0, 100.0, 110.0, 110.0]]), torch.tensor([0])),
    ]
    preds = [
        (torch.tensor([[100.0, 100.0, 110.0, 110.0]]), torch.tensor([0.9]), torch.tensor([0])),
        (torch.zeros(0, 4), torch.zeros(0), torch.zeros(0, dtype=torch.long)),
    ]

    ap = average_precision_per_class(preds, gts, class_id=0, iou_threshold=0.5)
    assert ap == 0.0  # if image isolation were broken, this would incorrectly score > 0


def test_mean_average_precision_skips_classes_absent_from_gt():
    gts = [(torch.tensor([[10.0, 10.0, 50.0, 50.0]]), torch.tensor([0]))]
    preds = [(torch.tensor([[10.0, 10.0, 50.0, 50.0]]), torch.tensor([0.9]), torch.tensor([0]))]

    results = mean_average_precision(preds, gts, num_classes=2, iou_thresholds=(0.5,))
    assert results[0.5] == 1.0  # class 1 has no GT anywhere, excluded from the mean


def test_mean_average_precision_averages_across_thresholds():
    gts = [(torch.tensor([[10.0, 10.0, 50.0, 50.0]]), torch.tensor([0]))]
    preds = [(torch.tensor([[10.0, 10.0, 50.0, 50.0]]), torch.tensor([0.9]), torch.tensor([0]))]

    results = mean_average_precision(preds, gts, num_classes=1, iou_thresholds=(0.5, 0.75))
    assert results[0.5] == 1.0
    assert results[0.75] == 1.0
    assert results["mAP@[.5:.95]"] == 1.0
