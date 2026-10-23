import argparse
import json
import os

from PIL import Image

import train_model as train_cli


def make_manifest(tmp_path):
    for name in ["img1.jpg", "img2.jpg"]:
        Image.new("RGB", (64, 64)).save(tmp_path / name)

    manifest = {
        "train": [
            {"image_path": str(tmp_path / "img1.jpg"), "boxes": [[5.0, 5.0, 20.0, 20.0]], "labels": [0]},
            {"image_path": str(tmp_path / "img2.jpg"), "boxes": [[10.0, 10.0, 15.0, 15.0]], "labels": [1]},
        ],
        "valid": [
            {"image_path": str(tmp_path / "img1.jpg"), "boxes": [[5.0, 5.0, 20.0, 20.0]], "labels": [0]},
        ],
        "test": [],
    }
    manifest_path = tmp_path / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)
    return str(manifest_path)


def make_args(manifest_path, out_dir, **overrides):
    defaults = dict(
        manifest=manifest_path, out_dir=out_dir, num_classes=2, img_size=64,
        strides=[8, 16, 32], thresholds=[16.0, 32.0], epochs=2, batch_size=2, lr=1e-3,
        warmup_epochs=1.0, final_lr_ratio=0.01, weight_decay=0.01, box_weight=5.0, center_radius=1,
        no_augment=False, aug_scale=[0.7, 1.3], aug_translate=0.1, num_workers=0,
        repeat_threshold=0.5, no_repeat_sampler=False,
        conf_thresh=0.3, iou_thresh=0.5, map_iou_thresholds=[0.5], map_every=1, device="cpu", seed=42,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_train_cli_runs_end_to_end_with_repeat_sampler(tmp_path):
    manifest_path = make_manifest(tmp_path)
    out_dir = str(tmp_path / "runs")

    train_cli.main(make_args(manifest_path, out_dir))

    assert os.path.exists(os.path.join(out_dir, "latest.pt"))
    assert os.path.exists(os.path.join(out_dir, "best.pt"))  # val manifest is non-empty


def test_train_cli_runs_without_repeat_sampler(tmp_path):
    manifest_path = make_manifest(tmp_path)
    out_dir = str(tmp_path / "runs")

    train_cli.main(make_args(manifest_path, out_dir, no_repeat_sampler=True))

    assert os.path.exists(os.path.join(out_dir, "latest.pt"))


def test_train_cli_forwards_eval_thresholds(tmp_path, monkeypatch):
    captured = {}
    monkeypatch.setattr(train_cli, "train", lambda *a, **kw: captured.update(kw))

    train_cli.main(make_args(make_manifest(tmp_path), str(tmp_path / "runs"),
                             conf_thresh=0.001, iou_thresh=0.6))

    assert captured["conf_thresh"] == 0.001
    assert captured["iou_thresh"] == 0.6


def test_train_cli_resumes_on_a_second_call(tmp_path):
    manifest_path = make_manifest(tmp_path)
    out_dir = str(tmp_path / "runs")

    train_cli.main(make_args(manifest_path, out_dir, epochs=1))
    train_cli.main(make_args(manifest_path, out_dir, epochs=2))

    from train.trainer import load_checkpoint
    from models.yolo import YOLO
    model = YOLO(2)
    state = load_checkpoint(os.path.join(out_dir, "latest.pt"), model, device="cpu")
    assert state["epoch"] == 1  # 2 epochs total, 0-indexed -> last saved is epoch 1
