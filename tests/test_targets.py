import torch

from train.targets import assign_level, build_targets, build_targets_batch, decode_boxes
from train.loss import yolo_loss

IMG_SIZE = 320
STRIDES = (8, 16, 32)
THRESHOLDS = (64, 128)
NUM_CLASSES = 5


def test_decode_boxes_clamps_negative_raw_predictions():
    # an undertrained model can output a negative (l,t,r,b) -- decode must never
    # produce an inverted box (x1 > x2 or y1 > y2), which downstream code (PIL,
    # IoU, NMS) can't handle
    box_pred = torch.tensor([[[-5.0]], [[-5.0]], [[-5.0]], [[-5.0]]]).unsqueeze(0)  # (1,4,1,1)
    decoded = decode_boxes(box_pred, stride=8)

    x1, y1, x2, y2 = decoded[0, 0, 0].tolist()
    assert x1 <= x2
    assert y1 <= y2


def test_assign_level_boundaries():
    assert assign_level(10, THRESHOLDS) == 0
    assert assign_level(64, THRESHOLDS) == 0  # inclusive upper bound
    assert assign_level(65, THRESHOLDS) == 1
    assert assign_level(128, THRESHOLDS) == 1
    assert assign_level(129, THRESHOLDS) == 2


def test_single_small_box_assigned_to_level0():
    boxes = torch.tensor([[100.0, 100.0, 144.0, 144.0]])  # size 44 -> level 0
    labels = torch.tensor([2])

    targets = build_targets(boxes, labels, IMG_SIZE, IMG_SIZE, STRIDES, NUM_CLASSES, THRESHOLDS)
    _, cls_t0 = targets[0]
    _, cls_t1 = targets[1]
    _, cls_t2 = targets[2]

    assert cls_t0.sum().item() == 1.0
    assert cls_t1.sum().item() == 0.0
    assert cls_t2.sum().item() == 0.0

    # center (122, 122), stride 8 -> cell (15, 15)
    assert cls_t0[2, 15, 15] == 1.0


def test_decode_recovers_original_box():
    boxes = torch.tensor([[100.0, 100.0, 144.0, 144.0]])
    labels = torch.tensor([0])
    targets = build_targets(boxes, labels, IMG_SIZE, IMG_SIZE, STRIDES, NUM_CLASSES, THRESHOLDS)
    box_t0, _ = targets[0]

    decoded = decode_boxes(box_t0.unsqueeze(0), STRIDES[0])
    x1, y1, x2, y2 = decoded[0, 15, 15].tolist()
    assert abs(x1 - 100.0) < 1e-4
    assert abs(y1 - 100.0) < 1e-4
    assert abs(x2 - 144.0) < 1e-4
    assert abs(y2 - 144.0) < 1e-4


def test_collision_keeps_smaller_box():
    boxes = torch.tensor([
        [100.0, 100.0, 144.0, 144.0],  # 44x44, area 1936
        [110.0, 110.0, 140.0, 140.0],  # 30x30, area 900, same cell at level 0
    ])
    labels = torch.tensor([0, 1])

    targets = build_targets(boxes, labels, IMG_SIZE, IMG_SIZE, STRIDES, NUM_CLASSES, THRESHOLDS)
    _, cls_t0 = targets[0]

    assert cls_t0.sum().item() == 1.0
    assert cls_t0[1].sum().item() == 1.0  # smaller box (label 1) won the cell
    assert cls_t0[0].sum().item() == 0.0


def test_batch_targets_stack_correctly():
    boxes_list = [
        torch.tensor([[100.0, 100.0, 144.0, 144.0]]),
        torch.tensor([[0.0, 0.0, 16.0, 16.0], [300.0, 300.0, 319.0, 319.0]]),
    ]
    labels_list = [torch.tensor([0]), torch.tensor([1, 2])]

    batched = build_targets_batch(
        boxes_list, labels_list, IMG_SIZE, IMG_SIZE, STRIDES, NUM_CLASSES, THRESHOLDS
    )

    assert len(batched) == 3
    for level, stride in enumerate(STRIDES):
        h = w = IMG_SIZE // stride
        box_targets, cls_targets = batched[level]
        assert box_targets.shape == (2, 4, h, w)
        assert cls_targets.shape == (2, NUM_CLASSES, h, w)


def test_loss_near_zero_on_perfect_prediction():
    boxes = torch.tensor([[100.0, 100.0, 144.0, 144.0]])
    labels = torch.tensor([2])
    single = build_targets(boxes, labels, IMG_SIZE, IMG_SIZE, STRIDES, NUM_CLASSES, THRESHOLDS)

    targets = [(bt.unsqueeze(0), ct.unsqueeze(0)) for bt, ct in single]
    preds = []
    for box_target, cls_target in targets:
        box_pred = box_target.clone()
        cls_pred = torch.where(cls_target == 1.0, torch.tensor(10.0), torch.tensor(-10.0))
        preds.append((box_pred, cls_pred))

    out = yolo_loss(preds, targets)
    assert out["box"].item() == 0.0
    assert out["cls"].item() < 1e-2


def test_loss_handles_no_boxes():
    targets, preds = [], []
    for stride in STRIDES:
        h = w = IMG_SIZE // stride
        targets.append((torch.zeros(1, 4, h, w), torch.zeros(1, NUM_CLASSES, h, w)))
        preds.append((torch.randn(1, 4, h, w), torch.full((1, NUM_CLASSES, h, w), -10.0)))

    out = yolo_loss(preds, targets)
    assert out["box"].item() == 0.0
    assert torch.isfinite(out["total"])
