import os

import pytest
from PIL import Image

from data.yolo_format import copies_per_image, to_yolo_lines, write_yolo_dataset


def parse(line):
    cls, cx, cy, w, h = line.split()
    return int(cls), float(cx), float(cy), float(w), float(h)


def test_to_yolo_lines_normalizes_center_and_size():
    # 200x100 image, box from (20, 10) to (60, 50): center (40, 30), size 40x40
    (line,) = to_yolo_lines([[20, 10, 60, 50]], [7], img_w=200, img_h=100)
    cls, cx, cy, w, h = parse(line)

    assert cls == 7
    assert cx == pytest.approx(40 / 200)
    assert cy == pytest.approx(30 / 100)
    assert w == pytest.approx(40 / 200)
    assert h == pytest.approx(40 / 100)


def test_to_yolo_lines_round_trips_back_to_pixels():
    boxes = [[13.0, 27.5, 88.0, 61.25], [0.0, 0.0, 200.0, 100.0]]
    lines = to_yolo_lines(boxes, [0, 1], img_w=200, img_h=100)

    for line, (x1, y1, x2, y2) in zip(lines, boxes):
        _, cx, cy, w, h = parse(line)
        assert (cx - w / 2) * 200 == pytest.approx(x1, abs=1e-3)
        assert (cy - h / 2) * 100 == pytest.approx(y1, abs=1e-3)
        assert (cx + w / 2) * 200 == pytest.approx(x2, abs=1e-3)
        assert (cy + h / 2) * 100 == pytest.approx(y2, abs=1e-3)


def test_to_yolo_lines_clips_boxes_past_the_border():
    (line,) = to_yolo_lines([[-10, 90, 50, 130]], [0], img_w=100, img_h=100)
    _, cx, cy, w, h = parse(line)

    # clipped to (0, 90)-(50, 100)
    assert (cx, cy, w, h) == pytest.approx((0.25, 0.95, 0.5, 0.1))
    assert all(0.0 <= v <= 1.0 for v in (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))


def test_to_yolo_lines_skips_boxes_entirely_outside():
    assert to_yolo_lines([[120, 10, 150, 40]], [0], img_w=100, img_h=100) == []


def make_manifest(tmp_path):
    src = tmp_path / "src"
    for folder in ["a", "b"]:
        (src / folder).mkdir(parents=True)
    # two images with the SAME file name from different Roboflow folders
    Image.new("RGB", (200, 100)).save(src / "a" / "img.jpg")
    Image.new("RGB", (200, 100)).save(src / "b" / "img.jpg")
    Image.new("RGB", (50, 50)).save(src / "a" / "other.jpg")

    return {
        "train": [
            {"image_path": str(src / "a" / "img.jpg"), "boxes": [[20, 10, 60, 50], [0, 0, 10, 10]], "labels": [1, 0]},
            {"image_path": str(src / "b" / "img.jpg"), "boxes": [[5, 5, 15, 15]], "labels": [2]},
        ],
        "valid": [
            {"image_path": str(src / "a" / "other.jpg"), "boxes": [[10, 10, 20, 20]], "labels": [0]},
        ],
        "test": [],
    }


def test_write_yolo_dataset_layout_and_labels(tmp_path):
    out = tmp_path / "yolo"
    write_yolo_dataset(make_manifest(tmp_path), ["c0", "c1", "c2"], str(out))

    train_images = sorted(os.listdir(out / "images" / "train"))
    train_labels = sorted(os.listdir(out / "labels" / "train"))
    assert train_images == ["000000_img.jpg", "000001_img.jpg"]  # same name, both kept
    assert train_labels == ["000000_img.txt", "000001_img.txt"]  # mirrors images/
    assert os.listdir(out / "images" / "val") == ["000000_other.jpg"]
    assert os.listdir(out / "images" / "test") == []

    lines = (out / "labels" / "train" / "000000_img.txt").read_text().splitlines()
    assert [parse(l)[0] for l in lines] == [1, 0]  # one line per box, classes in order
    assert parse(lines[0])[1:] == pytest.approx((0.2, 0.3, 0.2, 0.4))


def test_copies_per_image_matches_repeat_factors_on_average():
    assert copies_per_image([1.0, 2.0, 3.0]) == [1, 2, 3]  # whole factors: exact

    counts = copies_per_image([1.5] * 10000)
    assert set(counts) == {1, 2}
    assert abs(sum(counts) / len(counts) - 1.5) < 0.02  # expected count = factor

    assert copies_per_image([1.3] * 50, seed=7) == copies_per_image([1.3] * 50, seed=7)  # reproducible


def test_write_yolo_dataset_repeats_only_train_images(tmp_path):
    out = tmp_path / "yolo"
    write_yolo_dataset(make_manifest(tmp_path), ["c0", "c1", "c2"], str(out), train_repeat_factors=[1.0, 3.0])

    train_images = sorted(os.listdir(out / "images" / "train"))
    assert train_images == ["000000_img.jpg", "000001_img.jpg", "000001_rep1_img.jpg", "000001_rep2_img.jpg"]
    assert sorted(os.listdir(out / "labels" / "train")) == [n.replace(".jpg", ".txt") for n in train_images]

    original = (out / "labels" / "train" / "000001_img.txt").read_text()
    assert (out / "labels" / "train" / "000001_rep2_img.txt").read_text() == original
    assert os.listdir(out / "images" / "val") == ["000000_other.jpg"]  # never repeated


def test_write_yolo_dataset_data_yaml(tmp_path):
    yaml = pytest.importorskip("yaml")
    out = tmp_path / "yolo"
    data_yaml = write_yolo_dataset(make_manifest(tmp_path), ["c0", "c: tricky", 'q"uote'], str(out))

    config = yaml.safe_load(open(data_yaml))
    assert config["path"] == os.path.abspath(out)
    assert (config["train"], config["val"], config["test"]) == ("images/train", "images/val", "images/test")
    assert config["names"] == {0: "c0", 1: "c: tricky", 2: 'q"uote'}  # quoting survives
