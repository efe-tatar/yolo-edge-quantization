"""CLI for post-training static quantization (PTQ) of an exported ONNX model to INT8,
through ONNX Runtime. Weights are quantized from the model itself; activations need
their value ranges measured first, which is what the calibration images are for.

Calibrates on TRAINING images only -- quantization is part of building the model, so
val and test stay out of it.

Usage:
    python quantize_model.py --onnx model_fp32.onnx --out model_int8.onnx --manifest manifest.json
    python quantize_model.py --onnx model_fp32.onnx --out model_int8_fh.onnx --manifest manifest.json --float-heads
"""
import argparse
import json
import os
import random
import tempfile

import onnx
from onnxruntime.quantization import (
    CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType, quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process

from data.dataset import DetectionDataset


class CalibrationImages(CalibrationDataReader):
    """Feeds images one at a time, preprocessed exactly like inference: the same
    DetectionDataset path, with no augmentation."""

    def __init__(self, samples, img_size):
        self.dataset = DetectionDataset(samples, img_size=img_size)
        self.next_index = 0

    def get_next(self):
        if self.next_index >= len(self.dataset):
            return None
        image = self.dataset[self.next_index][0]
        self.next_index += 1
        return {"images": image.unsqueeze(0).numpy()}


def head_output_nodes(model_path):
    """Names of the final box/obj/cls convs: the nodes that directly produce the
    graph's outputs."""
    model = onnx.load(model_path)
    outputs = {o.name for o in model.graph.output}
    return [node.name for node in model.graph.node if outputs & set(node.output)]


def quantize(onnx_path, out_path, samples, img_size=320, float_heads=False,
             calibration_method="MinMax"):
    """float_heads: keep the final output convs in FP32. Each output is otherwise
    squeezed into 256 levels, which can cost accuracy where small differences matter
    (a box edge, a logit near the decision boundary), in exchange for slightly more
    compute."""
    # the pre-process step folds BatchNorm into Conv and infers shapes -- quantizing
    # without it leaves ranges measured at the wrong points of the graph
    with tempfile.TemporaryDirectory() as tmp:
        prepared = os.path.join(tmp, "prepared.onnx")
        # skip_symbolic_shape: ORT's symbolic inference can't complete on this graph;
        # ONNX's standard shape inference (still run) is enough for quantization
        quant_pre_process(onnx_path, prepared, skip_symbolic_shape=True)

        quantize_static(
            prepared, out_path, CalibrationImages(samples, img_size),
            quant_format=QuantFormat.QDQ, per_channel=True,
            activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
            calibrate_method=CalibrationMethod[calibration_method],
            nodes_to_exclude=head_output_nodes(prepared) if float_heads else None,
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True, help="FP32 model from export_model.py")
    parser.add_argument("--out", required=True)
    parser.add_argument("--manifest", required=True, help="manifest.json, for its train images")
    parser.add_argument("--img-size", type=int, default=320)
    parser.add_argument("--calibration-images", type=int, default=200)
    parser.add_argument("--calibration-method", choices=["MinMax", "Percentile"], default="MinMax")
    parser.add_argument("--float-heads", action="store_true",
                        help="keep the final box/obj/cls convs in FP32")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main(args):
    with open(args.manifest) as f:
        train_samples = json.load(f)["train"]
    n = min(args.calibration_images, len(train_samples))
    samples = random.Random(args.seed).sample(train_samples, n)

    quantize(args.onnx, args.out, samples, args.img_size, args.float_heads, args.calibration_method)
    print(f"quantized {args.onnx} -> {args.out}, calibrated on {n} train images")


if __name__ == "__main__":
    main(parse_args())
