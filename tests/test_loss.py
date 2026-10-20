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
