import torch
from PIL import Image

from data.dataset import DetectionDataset, collate_fn


def make_sample(tmp_path, file_name, w, h, boxes, labels):
    path = tmp_path / file_name
    Image.new("RGB", (w, h)).save(path)
    return {"image_path": str(path), "boxes": boxes, "labels": labels}


def test_dataset_scales_boxes_to_img_size(tmp_path):
    sample = make_sample(tmp_path, "img1.jpg", 100, 200, [[10, 20, 30, 40]], [0])
    ds = DetectionDataset([sample], img_size=320)

    assert len(ds) == 1
    image_tensor, boxes, labels = ds[0]
    assert image_tensor.shape == (3, 320, 320)
    assert boxes.shape == (1, 4)
    assert labels.tolist() == [0]

    sx, sy = 320 / 100, 320 / 200
    expected = torch.tensor([[10 * sx, 20 * sy, 30 * sx, 40 * sy]])
    assert torch.allclose(boxes, expected)


def test_dataset_keeps_negative_images_with_no_boxes(tmp_path):
    sample = make_sample(tmp_path, "img1.jpg", 100, 100, [], [])
    ds = DetectionDataset([sample], img_size=320)

    _, boxes, labels = ds[0]
    assert boxes.shape == (0, 4)
    assert labels.shape == (0,)


def test_collate_fn_stacks_images_and_keeps_boxes_ragged(tmp_path):
    s1 = make_sample(tmp_path, "img1.jpg", 100, 100, [[10, 10, 20, 20]], [0])
    s2 = make_sample(tmp_path, "img2.jpg", 100, 100, [[10, 10, 20, 20], [30, 30, 40, 40]], [0, 1])
    ds = DetectionDataset([s1, s2], img_size=320)

    images, boxes, labels = collate_fn([ds[0], ds[1]])
    assert images.shape == (2, 3, 320, 320)
    assert len(boxes) == 2
    assert boxes[0].shape[0] == 1
    assert boxes[1].shape[0] == 2
    assert len(labels) == 2
    assert labels[1].tolist() == [0, 1]
