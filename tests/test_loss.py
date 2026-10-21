import torch

from train.loss import focal_loss, yolo_loss


def test_focal_loss_downweights_easy_examples_more_than_hard_ones():
    # "easy": the model is already confidently correct. "hard": it's unsure.
    easy_correct_negative = torch.tensor([-10.0])
    hard_uncertain_negative = torch.tensor([0.0])
    target = torch.tensor([0.0])

    import torch.nn.functional as F
    bce_easy = F.binary_cross_entropy_with_logits(easy_correct_negative, target)
    bce_hard = F.binary_cross_entropy_with_logits(hard_uncertain_negative, target)

    focal_easy = focal_loss(easy_correct_negative, target)
    focal_hard = focal_loss(hard_uncertain_negative, target)

    # focal loss should shrink the easy example's loss MUCH more than the hard one's,
    # relative to plain BCE -- that's the entire point of the modulating factor
    easy_ratio = focal_easy.item() / bce_easy.item()
    hard_ratio = focal_hard.item() / bce_hard.item()
    assert easy_ratio < hard_ratio


def test_focal_loss_near_zero_on_confident_correct_predictions():
    logits = torch.tensor([10.0, -10.0])
    targets = torch.tensor([1.0, 0.0])
    assert focal_loss(logits, targets).item() < 1e-6


def test_cls_loss_ignores_negative_cells_entirely():
    # cell 0 is positive (correct, confident prediction); cell 1 is negative, but its
    # cls_pred screams the wrong thing (10.0) against a target of 0 -- if cls loss
    # looked at negative cells at all, that alone would blow the loss way past 1e-2.
    # It shouldn't be looked at: that's the entire point of decoupling cls from obj.
    box_pred = torch.zeros(1, 4, 1, 2)
    box_target = torch.zeros(1, 4, 1, 2)

    obj_pred = torch.tensor([[[[10.0, -10.0]]]])
    obj_target = torch.tensor([[[[1.0, 0.0]]]])

    cls_pred = torch.tensor([[[[10.0, 10.0]], [[-10.0, -10.0]], [[-10.0, -10.0]]]])
    cls_target = torch.tensor([[[[1.0, 0.0]], [[0.0, 0.0]], [[0.0, 0.0]]]])

    preds = [(box_pred, obj_pred, cls_pred)]
    targets = [(box_target, obj_target, cls_target)]

    out = yolo_loss(preds, targets)
    assert out["cls"].item() < 1e-2


def _two_positive_cells(num_classes=3):
    """2 positive cells + 1 negative, random cls logits -- no exact zeros anywhere."""
    torch.manual_seed(0)
    box = torch.zeros(1, 4, 1, 3)
    obj_target = torch.tensor([[[[1.0, 1.0, 0.0]]]])
    cls_pred = torch.randn(1, num_classes, 1, 3)
    cls_target = torch.zeros(1, num_classes, 1, 3)
    cls_target[0, 0, 0, 0] = 1.0
    cls_target[0, 1, 0, 1] = 1.0
    return [(box, torch.zeros_like(obj_target), cls_pred)], [(box, obj_target, cls_target)]


def test_cls_loss_per_positive_is_num_classes_times_per_element():
    preds, targets = _two_positive_cells(num_classes=3)

    per_positive = yolo_loss(preds, targets, cls_norm="positive")["cls"].item()
    per_element = yolo_loss(preds, targets, cls_norm="element")["cls"].item()

    assert abs(per_positive - 3 * per_element) < 1e-5


def test_cls_loss_per_positive_matches_hand_computation():
    import torch.nn.functional as F
    preds, targets = _two_positive_cells(num_classes=3)
    cls_pred, cls_target = preds[0][2], targets[0][2]

    # sum the BCE over every class at each positive cell, then average over the 2 cells
    expected = sum(
        F.binary_cross_entropy_with_logits(cls_pred[0, :, 0, i], cls_target[0, :, 0, i], reduction="sum")
        for i in (0, 1)
    ) / 2

    assert abs(yolo_loss(preds, targets)["cls"].item() - expected.item()) < 1e-5


def test_cls_norm_rejects_unknown_values():
    import pytest
    preds, targets = _two_positive_cells()
    with pytest.raises(ValueError):
        yolo_loss(preds, targets, cls_norm="mean")


def _obj_only(obj_pred, obj_target):
    """Zero box/cls so only the obj term varies between calls."""
    _, _, h, w = obj_pred.shape
    box = torch.zeros(1, 4, h, w)
    cls = torch.zeros(1, 1, h, w)
    return [(box, obj_pred, cls)], [(box, obj_target, cls)]


def test_balanced_obj_loss_matches_hand_computation():
    import torch.nn.functional as F
    obj_pred = torch.tensor([[[[2.0, -1.0, 0.5, -3.0]]]])
    obj_target = torch.tensor([[[[1.0, 0.0, 0.0, 0.0]]]])
    preds, targets = _obj_only(obj_pred, obj_target)

    bce = F.binary_cross_entropy_with_logits(obj_pred, obj_target, reduction="none").flatten()
    expected = 1.0 * bce[0] + 0.5 * bce[1:].mean()

    out = yolo_loss(preds, targets, obj_weight=1.0, noobj_weight=0.5, obj_norm="balanced")
    assert abs(out["obj"].item() - expected.item()) < 1e-6


def test_balanced_obj_loss_does_not_dilute_positives_with_more_empty_cells():
    # the same positive + identical negatives, once with 3 empty cells, once with 300.
    # What matters is the gradient pushing the positive's logit up: under "element" it
    # shrinks ~1/num_cells as the grid grows (the bug), under "balanced" it doesn't move
    def positive_grad(n_neg, obj_norm):
        obj_pred = torch.full((1, 1, 1, n_neg + 1), -2.0)
        obj_pred[0, 0, 0, 0] = 0.0
        obj_pred.requires_grad_(True)
        obj_target = torch.zeros_like(obj_pred)
        obj_target[0, 0, 0, 0] = 1.0
        preds, targets = _obj_only(obj_pred, obj_target)
        yolo_loss(preds, targets, obj_norm=obj_norm)["total"].backward()  # "obj" is detached (logging only)
        return obj_pred.grad[0, 0, 0, 0].abs().item()

    assert abs(positive_grad(3, "balanced") - positive_grad(300, "balanced")) < 1e-6
    assert positive_grad(300, "element") < positive_grad(3, "element") / 50


def test_balanced_obj_loss_handles_a_level_with_no_positives():
    obj_pred = torch.tensor([[[[-1.0, 1.0]]]])
    obj_target = torch.zeros_like(obj_pred)
    preds, targets = _obj_only(obj_pred, obj_target)

    out = yolo_loss(preds, targets, obj_norm="balanced")
    assert torch.isfinite(out["obj"]) and out["obj"].item() > 0


def test_obj_norm_rejects_unknown_values():
    import pytest
    preds, targets = _obj_only(torch.zeros(1, 1, 1, 2), torch.zeros(1, 1, 1, 2))
    with pytest.raises(ValueError):
        yolo_loss(preds, targets, obj_norm="mean")


def test_obj_loss_applies_noobj_weight_to_negative_cells():
    # same raw BCE magnitude at both cells (same logit, opposite target) -- a smaller
    # noobj_weight should shrink the negative cell's contribution to the average
    box_pred = torch.zeros(1, 4, 1, 2)
    box_target = torch.zeros(1, 4, 1, 2)
    cls_pred = torch.zeros(1, 1, 1, 2)
    cls_target = torch.zeros(1, 1, 1, 2)

    obj_pred = torch.tensor([[[[0.0, 0.0]]]])
    obj_target = torch.tensor([[[[1.0, 0.0]]]])  # cell 0 positive, cell 1 negative

    preds = [(box_pred, obj_pred, cls_pred)]
    targets = [(box_target, obj_target, cls_target)]

    weighted = yolo_loss(preds, targets, obj_weight=1.0, noobj_weight=0.5)
    unweighted = yolo_loss(preds, targets, obj_weight=1.0, noobj_weight=1.0)

    assert weighted["obj"].item() < unweighted["obj"].item()
