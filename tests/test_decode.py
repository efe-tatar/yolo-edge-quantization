import torch

from train.decode import decode_predictions

NUM_CLASSES = 3


def make_level(h, w, num_classes=NUM_CLASSES, batch=1):
    """All-negative obj_pred/cls_pred (nothing fires), zero box_pred -- a blank level."""
    box_pred = torch.zeros(batch, 4, h, w)
    obj_pred = torch.full((batch, 1, h, w), -10.0)
    cls_pred = torch.full((batch, num_classes, h, w), -10.0)
    return box_pred, obj_pred, cls_pred


def test_single_positive_cell_gives_one_box():
    box_pred, obj_pred, cls_pred = make_level(3, 3)
    obj_pred[0, 0, 1, 1] = 10.0
    cls_pred[0, 2, 1, 1] = 10.0  # class 2, cell (row=1, col=1)

    results = decode_predictions([(box_pred, obj_pred, cls_pred)], strides=(8,), conf_thresh=0.3)

    assert len(results) == 1  # one image in the batch
    boxes, scores, labels = results[0]
    assert boxes.shape == (1, 4)
    assert labels[0].item() == 2
    assert scores[0].item() > 0.99  # sigmoid(10) * sigmoid(10), both ~1


def test_all_negative_returns_empty_without_crashing():
    box_pred, obj_pred, cls_pred = make_level(3, 3)  # nothing fires anywhere

    results = decode_predictions([(box_pred, obj_pred, cls_pred)], strides=(8,), conf_thresh=0.3)

    boxes, scores, labels = results[0]
    assert boxes.shape == (0, 4)
    assert scores.shape == (0,)
    assert labels.shape == (0,)


def test_high_class_confidence_alone_is_not_enough_without_objectness():
    # a cell confidently guesses "class 2 if anything" but objectness says "nothing
    # here" -- the combined score (obj * cls) must stay below threshold
    box_pred, obj_pred, cls_pred = make_level(3, 3)
    cls_pred[0, 2, 1, 1] = 10.0  # obj_pred left at -10 (confidently "no object")

    results = decode_predictions([(box_pred, obj_pred, cls_pred)], strides=(8,), conf_thresh=0.3)

    boxes, _, _ = results[0]
    assert boxes.shape[0] == 0


def _make_overlapping_pair(class_a, class_b):
    """Two adjacent cells at stride 8, each with a wide box so they overlap heavily."""
    box_pred, obj_pred, cls_pred = make_level(1, 2)  # a 1x2 grid: cells (0,0) and (0,1)
    box_pred[0, :, 0, 0] = torch.tensor([2.0, 2.0, 2.0, 2.0])  # large (l,t,r,b), in stride units
    box_pred[0, :, 0, 1] = torch.tensor([2.0, 2.0, 2.0, 2.0])
    obj_pred[0, 0, 0, 0] = 10.0
    obj_pred[0, 0, 0, 1] = 10.0
    cls_pred[0, class_a, 0, 0] = 10.0
    cls_pred[0, class_b, 0, 1] = 10.0
    return box_pred, obj_pred, cls_pred


def test_nms_collapses_overlapping_boxes_of_the_same_class():
    box_pred, obj_pred, cls_pred = _make_overlapping_pair(class_a=1, class_b=1)

    results = decode_predictions(
        [(box_pred, obj_pred, cls_pred)], strides=(8,), conf_thresh=0.3, iou_thresh=0.5
    )

    boxes, scores, labels = results[0]
    assert boxes.shape[0] == 1  # the two heavily-overlapping boxes collapsed to one
    assert labels[0].item() == 1


def test_nms_keeps_overlapping_boxes_of_different_classes():
    box_pred, obj_pred, cls_pred = _make_overlapping_pair(class_a=0, class_b=1)

    results = decode_predictions(
        [(box_pred, obj_pred, cls_pred)], strides=(8,), conf_thresh=0.3, iou_thresh=0.5
    )

    boxes, scores, labels = results[0]
    assert boxes.shape[0] == 2  # batched_nms only suppresses within the same class
    assert set(labels.tolist()) == {0, 1}


def test_class_agnostic_nms_collapses_overlapping_boxes_of_different_classes():
    box_pred, obj_pred, cls_pred = _make_overlapping_pair(class_a=0, class_b=1)

    results = decode_predictions(
        [(box_pred, obj_pred, cls_pred)], strides=(8,), conf_thresh=0.3, iou_thresh=0.5,
        class_agnostic_nms=True,
    )

    boxes, scores, labels = results[0]
    assert boxes.shape[0] == 1  # one shared NMS group -> classes no longer protect each other


def test_merges_candidates_across_levels():
    box_pred_0, obj_pred_0, cls_pred_0 = make_level(3, 3)
    obj_pred_0[0, 0, 0, 0] = 10.0
    cls_pred_0[0, 0, 0, 0] = 10.0  # one positive cell at stride 8

    box_pred_1, obj_pred_1, cls_pred_1 = make_level(2, 2)
    obj_pred_1[0, 0, 1, 1] = 10.0
    cls_pred_1[0, 1, 1, 1] = 10.0  # one positive cell at stride 16, far away -> no overlap

    results = decode_predictions(
        [(box_pred_0, obj_pred_0, cls_pred_0), (box_pred_1, obj_pred_1, cls_pred_1)],
        strides=(8, 16),
        conf_thresh=0.3,
    )

    boxes, scores, labels = results[0]
    assert boxes.shape[0] == 2  # one box contributed by each level
    assert set(labels.tolist()) == {0, 1}


def test_batch_images_are_independent():
    box_pred, obj_pred, cls_pred = make_level(3, 3, batch=2)
    obj_pred[0, 0, 0, 0] = 10.0
    cls_pred[0, 0, 0, 0] = 10.0  # only image 0 has a positive cell
    # image 1 stays all-negative

    results = decode_predictions([(box_pred, obj_pred, cls_pred)], strides=(8,), conf_thresh=0.3)

    assert len(results) == 2
    boxes0, _, _ = results[0]
    boxes1, _, _ = results[1]
    assert boxes0.shape[0] == 1
    assert boxes1.shape[0] == 0
