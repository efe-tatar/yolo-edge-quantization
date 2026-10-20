# yolo-edge-quantization

A YOLO-style object detector, built from scratch in PyTorch, trained on a real
traffic-sign dataset, then quantized and benchmarked for CPU/edge deployment.

**Status: in progress.** Architecture, training pipeline, and dataset handling are
built and tested. No model has been trained yet -- see [Roadmap](#roadmap).

## Goal

The aim is to separate out, honestly, what each piece of a modern detector actually
buys you: the architecture itself, the training recipe, and pretraining. Building
the detector from scratch and comparing it against a reference implementation under
matched conditions is how this project tries to answer that:

1. **This repo's model**, trained from scratch.
2. **Ultralytics YOLOv8n**, trained from scratch on the same data.
3. **Ultralytics YOLOv8n**, fine-tuned from COCO-pretrained weights.

(1) vs (2) isolates how close a from-scratch reimplementation gets to a mature
reference architecture under identical conditions. (2) vs (3) isolates the value of
pretraining, which (1) can't replicate but can be measured against. The final
deliverable quantizes and benchmarks this repo's own model (PyTorch -> ONNX -> ONNX
Runtime, FP32 vs FP16 vs PTQ INT8 vs QAT INT8) on CPU.

## Architecture

Single-class-agnostic, multi-scale, anchor-free detector in the YOLOv8 family:

```
image --> Backbone --> P3, P4, P5 (strides 8/16/32)
                 \
                  --> Neck (FPN top-down + PAN bottom-up) --> 3 refined feature maps
                                  \
                                   --> Head x3 (one per scale) --> (box, cls) per cell
```

- **Backbone**: stride-2 `Conv` + `C2f` blocks, channel-doubling on every downsample,
  topped with `SPPF`.
- **Neck**: upsample + concat + `C2f` (top-down), then downsample + concat + `C2f`
  (bottom-up) -- lets every scale see both fine detail and deep semantics.
- **Head**: per scale, a shared conv trunk splits into a box-regression branch (4
  channels, `(l,t,r,b)` distances from the cell center) and a classification branch
  (`num_classes` channels). No separate objectness branch -- "is there an object" is
  implicit in the class scores, matching YOLOv8's actual design.
- **Target assignment**: FCOS-style static size-range routing (a box goes to the
  scale whose stride best matches its size), stride-normalized regression targets,
  smallest-box-wins on cell collisions.

## Repo structure

```
models/   Conv, Bottleneck, C2f, SPPF, Backbone, Neck, Head, YOLO
train/    target assignment, loss, inference decode + NMS, mAP, training loop
data/     dataset-agnostic: COCO-format pooling/redistribution/manifest, Dataset
colab/    dataset exploration/diagnostics (the only place that knows this is
          Roboflow's traffic-sign dataset specifically -- everything above is
          dataset-agnostic by design)
tests/    pytest, 65+ tests, every module above
```

## Dataset

[Roboflow: YOLO to COCO Traffic Signs](https://universe.roboflow.com/) (54 classes,
~6,000 images). The provided train/valid/test split isn't class-stratified, so
`data/coco_split.py` redistributes it photo-by-photo, rarest class first, 70/15/15 --
keeping every class (forced train-only below a minimum-photo threshold) rather than
dropping rare ones outright. See `colab/dataset_split.ipynb` for the full diagnostic.

## Running the tests

```
pip install -r requirements.txt   # torch, torchvision, pytest, Pillow, onnx, onnxruntime
python -m pytest tests/ -v
```

## Roadmap

- [x] Architecture (backbone, neck, head), fully tested
- [x] Target assignment, loss, inference decode + NMS, mAP, training loop
- [x] Dataset pooling/redistribution/manifest, verified against real data
- [x] Repeat-factor sampling (class imbalance is still severe even after the
      stratified split -- this is a training-time fix, not a splitting one)
- [x] ONNX export with a PyTorch-parity check
- [ ] A real training run
- [ ] Reference YOLOv8n baselines (from-scratch and pretrained)
- [ ] Three-way comparison
- [ ] Quantization (PTQ, QAT)
- [ ] CPU/edge benchmarking (latency, memory, accuracy-vs-latency Pareto)
