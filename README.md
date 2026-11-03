# yolo-edge-quantization

A YOLO-style object detector, built from scratch in PyTorch, trained on a real
traffic-sign dataset, then quantized and benchmarked for CPU/edge deployment.

**Result:** the from-scratch detector reaches **0.43 mAP@0.5 on test**, above YOLOv8n
trained from scratch (0.29) and below COCO-pretrained YOLOv8n (0.59). INT8 quantization
keeps mAP about level while making it **3.6x smaller and up to 2.2x faster on CPU**. See
[Comparison with YOLOv8n](#comparison-with-yolov8n) and [Quantization](#quantization).
Download the models from the
[v1.0 release](https://github.com/efe-tatar/yolo-edge-quantization/releases/tag/v1.0).

Getting a first working model took fixing three bugs that kept mAP at exactly 0 (see
[Debugging the first runs](#debugging-the-first-runs)); from there, val mAP@0.5 went from
0.156 to 0.402 over four runs (see [Training runs](#training-runs)).

## Goal

The aim is to separate out, honestly, what each piece of a modern detector actually
buys you: the architecture itself, the training recipe, and pretraining. Building
the detector from scratch and comparing it against a reference implementation under
matched conditions is how this project tries to answer that:

1. **This repo's model**, trained from scratch.
2. **Ultralytics YOLOv8n**, trained from scratch on the same data.
3. **Ultralytics YOLOv8n**, fine-tuned from COCO-pretrained weights.

(1) vs (2) measures how close a from-scratch reimplementation gets to a mature
reference architecture under matched conditions. (2) vs (3) isolates the value of
pretraining, which (1) can't replicate but can be measured against. The final
deliverable quantizes and benchmarks this repo's own model on CPU (PyTorch -> ONNX ->
ONNX Runtime, FP32 vs post-training-quantized INT8).

"Matched conditions" means matching what defines the task -- the data split, input
size (320), epoch budget, checkpoint selection on val, the scoring code, and anything
that changes what a label means (no horizontal flips: a mirrored "turn left" sign is
a "turn right" sign) -- while each model keeps its own training recipe and
post-processing, chosen on val. Final numbers come from the test split, used once.

## Architecture

Multi-class, multi-scale, anchor-free detector in the YOLOv8 family:

```
image --> Backbone --> P3, P4, P5 (strides 8/16/32)
                 \
                  --> Neck (FPN top-down + PAN bottom-up) --> 3 refined feature maps
                                  \
                                   --> Head x3 (one per scale) --> (box, obj, cls) per cell
```

- **Backbone**: stride-2 `Conv` + `C2f` blocks, channel-doubling on every downsample,
  topped with `SPPF`.
- **Neck**: upsample + concat + `C2f` (top-down), then downsample + concat + `C2f`
  (bottom-up) -- lets every scale see both fine detail and deep semantics.
- **Head**: per scale, a shared conv trunk splits into three branches -- box
  regression (4 channels, `(l,t,r,b)` distances from the cell center), objectness
  (1 channel), and classification (`num_classes` channels). Originally merged
  objectness into the class scores (YOLOv8's actual design), with focal loss to
  offset the resulting ~1:20,000 positive:negative cell ratio at train time; in
  practice this didn't converge within budget, so objectness was decoupled back out
  (YOLOv1-v5 style) -- classification loss is now computed only at positive cells,
  removing it from facing that imbalance at all, rather than just reweighting it.
  Caveat: the merged design was judged with an evaluation bug in place (bug 1 below),
  so it was never given a fair comparison.
- **Loss**: L1 box loss at positive cells; classification BCE summed over classes and
  normalized per positive cell; objectness BCE with positive and empty cells averaged
  separately. Both normalizations are fixes, explained in
  [Debugging the first runs](#debugging-the-first-runs).
- **Decode**: score = `sigmoid(obj) * max(sigmoid(cls))`, then per-class (or
  class-agnostic) NMS.
- **Target assignment**: FCOS-style static size-range routing (a box goes to the
  scale whose stride best matches its size), stride-normalized regression targets,
  smallest-box-wins on cell collisions, and center sampling: the cell containing the
  box center plus its neighbours whose centers fall inside the box are all positives
  (up to 9 per object, see [run03](results/run03_center/README.md)).

## Repo structure

```
models/    Conv, Bottleneck, C2f, SPPF, Backbone, Neck, Head, YOLO
train/     target assignment, loss, inference decode + NMS, mAP, training loop,
           per-branch diagnostics
data/      dataset-agnostic: COCO-format pooling/redistribution/manifest, Dataset,
           augmentation, export to Ultralytics' format (same split for the baselines)
colab/     dataset split, training (own model + YOLOv8n), debugging, and model
           comparison notebooks (the only place that knows this is Roboflow's
           traffic-sign dataset specifically -- everything above is dataset-agnostic)
results/   per-run metrics, logs and prediction images; the debugging and
           comparison write-ups
tests/     pytest, 131 tests, every module above
train_model.py, export_model.py, quantize_model.py, benchmark.py
           CLIs: training, ONNX export, INT8 quantization, CPU latency
```

## Dataset

[Roboflow: YOLO to COCO Traffic Signs](https://universe.roboflow.com/yolo-to-coco-ca2en/yolo-to-coco-kpv56) (54 classes,
~6,000 images). The provided train/valid/test split isn't class-stratified, so
`data/coco_split.py` redistributes it photo-by-photo, rarest class first, 70/15/15 --
keeping every class (forced train-only below a minimum-photo threshold) rather than
dropping rare ones outright. See `colab/dataset_split.ipynb` for the full diagnostic.

## Debugging the first runs

Every early run reported val mAP@0.5 = 0.0000 while all losses kept falling. Three
separate bugs were behind it. Each was found by measuring one branch of the head in
isolation (`train/diagnostics.py`, class-agnostic mAP), and each fix was confirmed by
an experiment where it was the only change. Full numbers and plots:
[results/debugging](results/debugging/README.md).

1. **The evaluation threshold hid real detections.** Predictions were filtered at
   confidence 0.3 before computing mAP, and early in training every true positive
   scored below that. At 0.001, the same checkpoint scored 0.005 mAP -- and **0.265
   class-agnostic mAP**: boxes were already good, classes weren't.
2. **The classification loss was scaled down 54x.** BCE averaged over (positive cells
   x 54 classes) gave the correct class 1/54 of a gradient. The classifier collapsed
   onto the most frequent class (59% of predictions for a class that's 12.5% of
   objects), equally on train and val. Normalizing per positive cell (as YOLOv8 does)
   took top-1 on a 64-image overfit test from 0.30 to 1.00.
3. **Objectness was drowned by empty cells.** One mean over ~2,100 cells per image,
   1-2 of them positive, pushed objectness to ~0 everywhere (median 0.002 at real
   objects). Averaging positive and empty cells separately took the overfit test's
   mAP from 0.26 to **0.97**.

Bugs 2 and 3 are the same mistake: averaging a loss over many elements when only a
few of them carry the signal.

## Training runs

This repo's model, trained from scratch: 320px input, same data split, scored with this
repo's own mAP code on the val split. Each run changes one thing from the one before;
its folder has the config, full log, prediction images and observations.

| run | change | epochs | best val mAP@0.5 | what it showed |
|---|---|---|---|---|
| [run01_baseline](results/run01_baseline/README.md) | -- | 50 | 0.156 | classification overfits from epoch 18; stacked duplicate boxes |
| [run02_augment](results/run02_augment/README.md) | + augmentation (no flips), AdamW, warmup + cosine LR | 50 | 0.254 | overfitting gone; val mAP still rising at the end |
| [run03_center](results/run03_center/README.md) | + center sampling | 50 | 0.293 | duplicate stacks gone: neighbouring cells now learn the right class and box |
| [run04_long](results/run04_long/README.md) | trained 2x longer | 100 | **0.402** | plateau at ~0.40; late overfitting shows up as boxes on pictograms inside signs |

![run 4 predictions on test images](results/run04_long/conf07.png)

*Run 4 on test images, conf 0.7. Green: ground truth. Red: predictions.*

## Comparison with YOLOv8n

All on the test split, same data split, 320px, 100 epochs, scored by the same code
(`colab/compare_models.ipynb`). Full tables (per-class AP, recall by object size,
protocol): [results/comparison](results/comparison/README.md).

| | own (run04) | own INT8 | YOLOv8n scratch | YOLOv8n pretrained |
|---|---|---|---|---|
| parameters / GFLOPs at 320 | 3.57M / 1.92 | same | 3.02M / 2.03 | 3.02M / 2.03 |
| **mAP@0.5** | 0.431 | 0.438 | 0.293 | **0.591** |
| mAP@[.5:.95] | 0.320 | 0.311 | 0.232 | **0.507** |
| class-agnostic mAP@0.5 | 0.680 | 0.662 | 0.764 | **0.927** |
| F1 at each model's threshold from val | 0.626 | 0.607 | 0.584 | **0.796** |

What it shows:

- **Pretraining is the largest effect measured.** Same architecture and recipe, and COCO
  weights take YOLOv8n from 0.293 to 0.591.
- **Against YOLOv8n from scratch, this model classifies better and localizes worse.**
  YOLOv8n leads when classes are ignored (0.764 vs 0.680) and this model leads once they
  count (0.431 vs 0.293). That's consistent with this model's simpler L1 box loss and fixed
  size ranges against DFL + CIoU and dynamic assignment, though no run isolated those.
- **This model's confidence ranks its detections poorly.** On 32-64 px objects it can reach
  93% recall at conf 0.001 but 65% at its threshold of 0.97, where pretrained goes from 98%
  to 94%. Duplicate boxes on one sign survive even at 0.97 (third row below).
- **A hypothesis that didn't hold.** On val, the gap to YOLOv8n's default recipe sat in rare
  classes and left/right signs, pointing at its horizontal flips and lack of class
  rebalancing. Removing the flips and adding repeat-factor sampling changed almost nothing
  (0.283 -> 0.284), so why YOLOv8n from scratch does this poorly remains unexplained.

**Caveats:** this model's recipe was tuned over four runs on val while YOLOv8n used
Ultralytics' defaults, so this isn't evidence of a better architecture. There's one
training run per model, and 27 of the 53 scored classes have fewer than 10 test objects.

![the four models on the same test images](results/comparison/side_by_side_test.png)

*Green: ground truth. Red: predictions, at each model's own threshold. Four images,
illustrative only.*

## Quantization

ONNX Runtime static post-training quantization of run04: QDQ, per-channel weights, MinMax
calibration on 200 training images. Latency on a laptop CPU (Intel i5-1335U), 320x320,
model inference only (no NMS), 100 runs.

| | FP32 | INT8 | change |
|---|---|---|---|
| ONNX size | 14.34 MB | **3.96 MB** | 3.6x smaller |
| median latency, 1 thread | 17.7 ms | **8.1 ms** | 2.2x faster |
| median latency, 2 threads | 9.5 ms | **5.8 ms** | 1.6x faster |
| median latency, 4 threads | 8.0 ms | **7.1 ms** | 1.1x faster |
| mAP@0.5 (test) | 0.431 | 0.438 | +0.006 |
| class-agnostic mAP@0.5 (test) | 0.680 | 0.662 | -0.017 |
| F1 (test) | 0.626 | 0.607 | -0.019 |

Accuracy is about level rather than better: the mAP@0.5 gain is too small to read
anything into, with INT8 ahead on 18 classes and behind on 20, and the other measures dip
slightly. These timings come from one sitting on a laptop, not an edge device, so the
ratios are safer to quote than the milliseconds.

## Usage

```
pip install -r requirements.txt
python -m pytest tests/

# manifest.json: built by data/coco_split.py (see colab/train_own_model.ipynb)
python train_model.py --manifest manifest.json --out-dir runs/own --num-classes 54 --epochs 100
python export_model.py --checkpoint runs/own/best.pt --num-classes 54 --out model_fp32.onnx
python quantize_model.py --onnx model_fp32.onnx --out model_int8.onnx --manifest manifest.json
python benchmark.py fp32=model_fp32.onnx int8=model_int8.onnx --threads 1 2 4
```

Training ran on Colab: `colab/train_own_model.ipynb` for this model,
`colab/train_yolov8n.ipynb` for the baselines, `colab/quantize_own_model.ipynb` for the
INT8 model. `colab/compare_models.ipynb` evaluates them all, on the same images.

Trained weights are in the
[v1.0 release](https://github.com/efe-tatar/yolo-edge-quantization/releases/tag/v1.0):
`model_int8.onnx`, `best.pt` (PyTorch checkpoint) and `classes.json` (class names). For the
INT8 model, use 320x320 input, confidence 0.95 and NMS IoU 0.5. The ONNX file doesn't
include box decoding or NMS; see `predict_onnx` in `train/analysis.py`.

## Roadmap

- [x] Architecture (backbone, neck, head), fully tested
- [x] Target assignment, loss, inference decode + NMS, mAP, training loop
- [x] Dataset pooling/redistribution/manifest, verified against real data
- [x] Repeat-factor sampling (class imbalance is still severe even after the
      stratified split -- this is a training-time fix, not a splitting one)
- [x] ONNX export with a PyTorch-parity check
- [x] Per-branch diagnostics; fixed eval threshold, cls and obj loss normalization
- [x] A real training run (baseline: val mAP@0.5 0.156)
- [x] Augmentation (box-aware, no horizontal flips) + LR schedule (val mAP@0.5 0.254)
- [x] Center sampling (several positive cells per object) (val mAP@0.5 0.293)
- [x] Long final training run with the settled recipe (100 epochs, val mAP@0.5 0.402)
- [x] Comparison tooling: per-class AP vs. training examples, class-agnostic mAP,
      recall by size, confusions, per-model NMS/threshold tuning
- [x] YOLOv8n from scratch (default recipe, then no flips + repeat-factor sampling)
- [x] YOLOv8n fine-tuned from COCO weights
- [x] Post-training INT8 quantization + CPU latency tooling
- [x] Final comparison of all four models on the test split (test mAP@0.5 0.431)
- [x] Quantization results (3.6x smaller, up to 2.2x faster, mAP about level)

## License

[MIT](LICENSE) for this repo's code. The YOLOv8n baselines are trained with
[Ultralytics](https://github.com/ultralytics/ultralytics) as a separate dependency,
under its own AGPL-3.0 license; none of its code or weights are included here.
