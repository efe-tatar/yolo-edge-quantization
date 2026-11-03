import argparse
import json
import os

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch
from PIL import Image

import benchmark
import export_model
import quantize_model
from data.dataset import DetectionDataset
from models.yolo import YOLO
from train.analysis import predict_onnx, predict_own_model

NUM_CLASSES = 3
IMG_SIZE = 64


def gradient_image(path, seed):
    pixels = np.random.default_rng(seed).integers(0, 255, (80, 100, 3), dtype=np.uint8)
    Image.fromarray(pixels).save(path)


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    """A real exported model + images to calibrate on, shared by the tests below."""
    tmp = tmp_path_factory.mktemp("quant")
    samples = []
    for i in range(4):
        path = str(tmp / f"img{i}.png")
        gradient_image(path, i)
        samples.append({"image_path": path, "boxes": [[10.0, 10.0, 40.0, 40.0]], "labels": [i % NUM_CLASSES]})

    # without BatchNorm statistics a fresh model's outputs are nearly constant, which
    # would make any quantized-vs-float comparison pass trivially
    torch.manual_seed(0)
    model = YOLO(NUM_CLASSES).train()
    images = torch.stack([DetectionDataset(samples, img_size=IMG_SIZE)[i][0] for i in range(4)])
    with torch.no_grad():
        for _ in range(20):
            model(images)
    model.eval()

    fp32 = str(tmp / "fp32.onnx")
    export_model.export(model, IMG_SIZE, fp32)
    return model, fp32, samples, tmp


def session(path):
    return ort.InferenceSession(path, providers=["CPUExecutionProvider"])


def test_quantize_produces_a_smaller_model_with_the_same_outputs(exported):
    _, fp32, samples, tmp = exported
    int8 = str(tmp / "int8.onnx")
    quantize_model.quantize(fp32, int8, samples, IMG_SIZE)

    assert os.path.getsize(int8) < os.path.getsize(fp32) / 2
    op_types = {node.op_type for node in onnx.load(int8).graph.node}
    assert {"QuantizeLinear", "DequantizeLinear"} <= op_types

    assert [o.name for o in session(int8).get_outputs()] == export_model.output_names()


def test_quantized_outputs_stay_close_to_fp32(exported):
    _, fp32, samples, tmp = exported
    int8 = str(tmp / "int8_close.onnx")
    quantize_model.quantize(fp32, int8, samples, IMG_SIZE)

    # one of the calibration images, so its values are inside the calibrated ranges
    image = quantize_model.CalibrationImages(samples, IMG_SIZE).get_next()
    for a, b in zip(session(fp32).run(None, image), session(int8).run(None, image)):
        assert a.max() - a.min() > 1e-3  # the outputs really vary, or this tests nothing
        assert np.abs(a - b).max() < 0.25 * (a.max() - a.min())  # distorted a bit, not scrambled


def test_float_heads_keeps_the_output_convs_unquantized(exported):
    _, fp32, samples, tmp = exported
    int8 = str(tmp / "int8_fh.onnx")
    quantize_model.quantize(fp32, int8, samples, IMG_SIZE, float_heads=True)

    graph = onnx.load(int8).graph
    outputs = {o.name for o in graph.output}
    producers = [node for node in graph.node if outputs & set(node.output)]
    assert len(producers) == len(outputs)
    assert {node.op_type for node in producers} == {"Conv"}  # not a DequantizeLinear


def test_calibration_reader_ends_and_matches_export_input(exported):
    _, _, samples, _ = exported
    reader = quantize_model.CalibrationImages(samples, IMG_SIZE)

    seen = 0
    while (batch := reader.get_next()) is not None:
        assert batch["images"].shape == (1, 3, IMG_SIZE, IMG_SIZE)
        assert batch["images"].dtype == np.float32
        seen += 1
    assert seen == len(samples)


def test_predict_onnx_matches_predict_own_model(exported):
    """A stub session returning the model's exact outputs, flattened as the export does:
    anything that differs is predict_onnx's own doing, not float noise (real ONNX
    parity is covered by tests/test_export.py)."""
    model, _, samples, _ = exported

    class ExactSession:
        def run(self, _, feed):
            with torch.no_grad():
                return [t.numpy() for level in model(torch.from_numpy(feed["images"])) for t in level]

    onnx_preds = predict_onnx(ExactSession(), samples, IMG_SIZE, conf_thresh=0.01, batch_size=3)
    torch_preds = predict_own_model(model, samples, IMG_SIZE, "cpu", conf_thresh=0.01, batch_size=3)

    assert len(onnx_preds) == len(samples)
    for pred_a, pred_b in zip(onnx_preds, torch_preds):
        for a, b in zip(pred_a, pred_b):
            assert torch.equal(a, b)


def test_cli_quantizes_from_a_manifest(exported):
    _, fp32, samples, tmp = exported
    manifest = str(tmp / "manifest.json")
    with open(manifest, "w") as f:
        json.dump({"train": samples, "valid": [], "test": []}, f)
    out = str(tmp / "cli.onnx")

    quantize_model.main(argparse.Namespace(onnx=fp32, out=out, manifest=manifest, img_size=IMG_SIZE,
                                           calibration_images=2, calibration_method="MinMax",
                                           float_heads=False, seed=0))
    session(out)  # loads


def test_benchmark_reports_one_row_per_model_and_thread_count(exported):
    _, fp32, _, _ = exported
    rows = benchmark.benchmark({"a": fp32, "b": fp32}, [1, 2], img_size=IMG_SIZE, runs=2)

    assert [(r["model"], r["threads"]) for r in rows] == [("a", 1), ("a", 2), ("b", 1), ("b", 2)]
    assert all(r["size_mb"] > 0 and r["median_ms"] > 0 and r["p95_ms"] >= r["median_ms"] * 0.5 for r in rows)
