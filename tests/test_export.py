import argparse
import os

import torch

import export_model
from models.yolo import YOLO
from train.trainer import save_checkpoint

NUM_CLASSES = 3
IMG_SIZE = 64


def make_checkpoint(tmp_path):
    model = YOLO(NUM_CLASSES)
    optimizer = torch.optim.Adam(model.parameters())
    path = str(tmp_path / "best.pt")
    save_checkpoint(path, model, optimizer, epoch=0)
    return path


def test_export_produces_a_loadable_onnx_file(tmp_path):
    checkpoint = make_checkpoint(tmp_path)
    out_path = str(tmp_path / "model.onnx")

    args = argparse.Namespace(
        checkpoint=checkpoint, num_classes=NUM_CLASSES, img_size=IMG_SIZE,
        out=out_path, opset=17,
    )
    export_model.main(args)

    assert os.path.exists(out_path)


def test_export_parity_within_tolerance(tmp_path):
    checkpoint = make_checkpoint(tmp_path)
    out_path = str(tmp_path / "model.onnx")

    model = YOLO(NUM_CLASSES)
    from train.trainer import load_checkpoint
    load_checkpoint(checkpoint, model, device="cpu")

    export_model.export(model, IMG_SIZE, out_path)
    max_diff = export_model.check_parity(model, IMG_SIZE, out_path)

    assert max_diff < 1e-4
