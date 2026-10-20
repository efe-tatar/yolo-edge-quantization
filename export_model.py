"""CLI for exporting a trained checkpoint to ONNX, with a parity check against the
original PyTorch model's output. Named export_model.py for the same reason as
train_model.py -- consistent naming, and avoids any future package-name collision.

Usage:
    python export_model.py --checkpoint runs/own_model/best.pt --num-classes 54 --out model.onnx
"""
import argparse

import numpy as np
import torch

from models.yolo import YOLO
from train.trainer import load_checkpoint


class ExportWrapper(torch.nn.Module):
    """YOLO.forward returns a nested ((box,cls), (box,cls), (box,cls)) tuple -- fine
    for training/eval code, but torch.onnx.export wants a flat tuple of tensors
    with named outputs. This wrapper only exists for export; it never touches the
    real model's forward signature, so nothing elsewhere has to change."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        outputs = self.model(x)
        return tuple(t for pair in outputs for t in pair)  # box0, cls0, box1, cls1, box2, cls2


def output_names(n_levels=3):
    names = []
    for i in range(n_levels):
        names += [f"box_{i}", f"cls_{i}"]
    return names


def export(model, img_size, out_path, opset=17):
    wrapped = ExportWrapper(model)
    wrapped.eval()  # must be called on the actual module passed to export, not just `model`
    dummy = torch.randn(1, 3, img_size, img_size)
    names = output_names()

    dynamic_axes = {"images": {0: "batch"}}
    dynamic_axes.update({name: {0: "batch"} for name in names})

    # dynamo=False: the older TorchScript-based exporter. The new default ("dynamo")
    # exporter in this torch version emits a Split op with a num_outputs attribute
    # that the installed onnxruntime build doesn't recognize -- a bleeding-edge
    # version mismatch between torch's exporter and onnxruntime, not a modeling issue.
    torch.onnx.export(
        wrapped, dummy, out_path,
        input_names=["images"], output_names=names,
        dynamic_axes=dynamic_axes, opset_version=opset, dynamo=False,
    )


def check_parity(model, img_size, onnx_path, atol=1e-4):
    import onnxruntime as ort

    model.eval()
    dummy = torch.randn(1, 3, img_size, img_size)
    with torch.no_grad():
        torch_outputs = model(dummy)
    torch_flat = [t.numpy() for pair in torch_outputs for t in pair]

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    onnx_outputs = session.run(None, {"images": dummy.numpy()})

    max_diff = max(np.abs(t - o).max() for t, o in zip(torch_flat, onnx_outputs))
    assert max_diff < atol, f"ONNX output diverges from PyTorch by {max_diff} (> {atol})"
    return float(max_diff)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num-classes", type=int, required=True)
    parser.add_argument("--img-size", type=int, default=320)
    parser.add_argument("--out", required=True)
    parser.add_argument("--opset", type=int, default=17)
    return parser.parse_args()


def main(args):
    model = YOLO(args.num_classes)
    load_checkpoint(args.checkpoint, model, device="cpu")

    export(model, args.img_size, args.out, args.opset)
    max_diff = check_parity(model, args.img_size, args.out)
    print(f"exported to {args.out}, max abs diff vs PyTorch: {max_diff:.2e}")


if __name__ == "__main__":
    main(parse_args())
