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


def test_yolo_loss_use_focal_toggle_falls_back_to_plain_bce():
    import torch.nn.functional as F

    cls_pred = torch.tensor([[[[0.0]]]])  # (1,1,1,1) -- uncertain logit, where focal differs from BCE
    cls_target = torch.tensor([[[[1.0]]]])
    box_pred = torch.zeros(1, 4, 1, 1)
    box_target = torch.zeros(1, 4, 1, 1)

    preds = [(box_pred, cls_pred)]
    targets = [(box_target, cls_target)]

    plain = yolo_loss(preds, targets, use_focal=False)
    expected_bce = F.binary_cross_entropy_with_logits(cls_pred, cls_target)
    assert torch.isclose(plain["cls"], expected_bce)

    focal = yolo_loss(preds, targets, use_focal=True)
    assert not torch.isclose(focal["cls"], expected_bce)  # focal actually changes something
