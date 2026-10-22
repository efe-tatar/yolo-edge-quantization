import torch
from PIL import Image

from data.augment import FILL, random_scale_translate, scale_translate
from data.dataset import DetectionDataset

SIZE = 100


def make_image():
    return Image.new("RGB", (SIZE, SIZE), (255, 0, 0))


def test_scale_translate_moves_boxes_with_the_image():
    boxes = torch.tensor([[10.0, 20.0, 30.0, 40.0]])
    labels = torch.tensor([3])

    image, out_boxes, out_labels = scale_translate(make_image(), boxes, labels, scale=0.5, offset=(5, 7))

    assert image.size == (SIZE, SIZE)
    assert torch.allclose(out_boxes, torch.tensor([[10 * 0.5 + 5, 20 * 0.5 + 7, 30 * 0.5 + 5, 40 * 0.5 + 7]]))
    assert out_labels.tolist() == [3]
    # zoomed out: the image content is red where it landed, grey outside
    assert image.getpixel((10, 10)) == (255, 0, 0)
    assert image.getpixel((90, 90)) == FILL


def test_scale_translate_clips_partially_visible_boxes():
    # zoom 2x, no shift: box (40..60) -> (80..120), half of it falls off the 100px canvas
    boxes = torch.tensor([[40.0, 40.0, 60.0, 60.0]])
    _, out_boxes, _ = scale_translate(make_image(), boxes, torch.tensor([0]), scale=2.0, offset=(0, 0),
                                      min_visible=0.2)

    assert torch.allclose(out_boxes, torch.tensor([[80.0, 80.0, 100.0, 100.0]]))


def test_scale_translate_drops_boxes_mostly_out_of_frame():
    boxes = torch.tensor([[10.0, 10.0, 20.0, 20.0], [85.0, 85.0, 95.0, 95.0]])
    labels = torch.tensor([1, 2])

    # shift right/down by 10: the second box is now 95..105 -> only 1/4 visible
    _, out_boxes, out_labels = scale_translate(make_image(), boxes, labels, scale=1.0, offset=(10, 10),
                                               min_visible=0.4)

    assert out_labels.tolist() == [1]  # labels are dropped together with their boxes
    assert out_boxes.shape == (1, 4)


def test_scale_translate_handles_images_without_boxes():
    image, out_boxes, out_labels = scale_translate(
        make_image(), torch.zeros((0, 4)), torch.zeros((0,), dtype=torch.long), scale=1.2, offset=(-5, -5)
    )
    assert image.size == (SIZE, SIZE)
    assert out_boxes.shape == (0, 4)
    assert out_labels.shape == (0,)


def test_random_scale_translate_keeps_boxes_inside_the_canvas():
    torch.manual_seed(0)
    boxes = torch.tensor([[0.0, 0.0, 50.0, 50.0], [40.0, 40.0, 100.0, 100.0]])
    labels = torch.tensor([0, 1])

    for _ in range(50):
        image, out_boxes, out_labels = random_scale_translate(make_image(), boxes, labels)
        assert image.size == (SIZE, SIZE)
        assert out_boxes.shape[0] == out_labels.shape[0]
        assert (out_boxes >= 0).all() and (out_boxes <= SIZE).all()
        assert (out_boxes[:, 2] > out_boxes[:, 0]).all() and (out_boxes[:, 3] > out_boxes[:, 1]).all()


def test_dataset_augment_only_when_enabled(tmp_path):
    path = tmp_path / "img.png"
    Image.new("RGB", (SIZE, SIZE), (0, 0, 255)).save(path)
    sample = {"image_path": str(path), "boxes": [[10, 10, 30, 30]], "labels": [0]}

    plain = DetectionDataset([sample], img_size=64)
    _, boxes_a, _ = plain[0]
    _, boxes_b, _ = plain[0]
    assert torch.equal(boxes_a, boxes_b)  # deterministic without augment

    torch.manual_seed(0)
    augmented = DetectionDataset([sample], img_size=64, augment=True)
    results = [augmented[0][1] for _ in range(10)]
    assert any(not torch.allclose(r, boxes_a) for r in results if r.shape == boxes_a.shape)
    assert all(augmented[0][0].shape == (3, 64, 64) for _ in range(3))
