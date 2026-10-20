import torch
from PIL import Image

from train.visualize import draw_detections


def test_returns_same_size_image_without_mutating_input():
    image = Image.new("RGB", (64, 64), color="white")
    boxes = torch.tensor([[10.0, 10.0, 30.0, 30.0]])
    labels = torch.tensor([0])

    out = draw_detections(image, boxes, labels)

    assert out.size == image.size
    assert image.tobytes() == Image.new("RGB", (64, 64), color="white").tobytes()  # input untouched


def test_zero_boxes_returns_an_unchanged_copy():
    image = Image.new("RGB", (64, 64), color="white")
    out = draw_detections(image, torch.zeros((0, 4)), torch.zeros((0,), dtype=torch.long))
    assert out.tobytes() == image.tobytes()


def test_drawing_a_box_actually_changes_pixels():
    image = Image.new("RGB", (64, 64), color="white")
    boxes = torch.tensor([[10.0, 10.0, 30.0, 30.0]])
    labels = torch.tensor([0])

    out = draw_detections(image, boxes, labels)
    assert out.tobytes() != image.tobytes()


def test_works_without_scores_for_ground_truth():
    image = Image.new("RGB", (64, 64), color="white")
    boxes = torch.tensor([[10.0, 10.0, 30.0, 30.0]])
    labels = torch.tensor([0])

    out = draw_detections(image, boxes, labels, scores=None, class_names=["sign"], color="green")
    assert out.size == image.size


def test_works_with_scores_and_class_names_for_predictions():
    image = Image.new("RGB", (64, 64), color="white")
    boxes = torch.tensor([[10.0, 10.0, 30.0, 30.0]])
    labels = torch.tensor([0])
    scores = torch.tensor([0.87])

    out = draw_detections(image, boxes, labels, scores=scores, class_names=["sign"])
    assert out.size == image.size
