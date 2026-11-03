# Comparison with Ultralytics YOLOv8n

This repo's model (run04) against Ultralytics YOLOv8n trained from scratch on the
same data. Produced with `colab/compare_models.ipynb`; all models are scored by the
same code (`train/metrics.py`, `train/analysis.py`) on the same images.

**All numbers here are on the val split.** The own model's recipe was developed
against val over four runs, so its val numbers are slightly optimistic; the final
numbers will come from the test split, once, at the end. YOLOv8n pretrained is still
to be trained.

## Protocol

| matched (defines the task) | each model's own (defines the method) |
|---|---|
| same data split (the saved `manifest.json`) | training recipe (augmentation, optimizer, LR schedule, losses) |
| input size 320 | resizing: Ultralytics letterboxes, the own model stretches |
| 100-epoch budget, best checkpoint chosen on val | post-processing: NMS IoU tuned per model on val (0.5 / 0.6 / 0.7) |
| no horizontal flips, repeat-factor sampling (second YOLOv8n run only, see below) | confidence operating point: each model's own best-F1 threshold |
| scoring code: mAP at conf 0.001, same matching | |

YOLOv8n was trained twice:

- **default**: Ultralytics' default recipe, unchanged. That recipe mirrors images
  horizontally (`fliplr=0.5`), which turns a "turn left" sign into a "turn right" sign
  with the wrong label, and doesn't rebalance rare classes.
- **no-flip + rfs**: `fliplr=0.0`, plus repeat-factor sampling with the own model's
  factors (rare-class train images written several times, see
  `data/yolo_format.py`). Everything else unchanged.

## Results

| | own (run04) | YOLOv8n default | YOLOv8n no-flip + rfs |
|---|---|---|---|
| parameters | 3.57M | 3.02M | 3.02M |
| GFLOPs at 320x320 | 1.92 | 2.03 | 2.03 |
| best NMS IoU | 0.5 | 0.5 | 0.5 |
| **mAP@0.5** | **0.402** | 0.283 | 0.284 |
| class-agnostic mAP@0.5 | 0.678 | **0.769** | 0.759 |
| mAP lost to wrong classes | **0.276** | 0.486 | 0.475 |
| best-F1 confidence | 0.97 | 0.30 | 0.40 |
| precision / recall at best F1 | **0.853** / 0.494 | 0.693 / 0.491 | 0.800 / 0.473 |
| F1 | **0.626** | 0.575 | 0.594 |

*Class-agnostic mAP: every label set to one class -- "found the object, box in the
right place". Its gap to mAP@0.5 is what classification mistakes cost.*

NMS IoU barely matters for either model (own: 0.402 / 0.399 / 0.392 at 0.5 / 0.6 / 0.7;
YOLOv8n default: 0.283 / 0.283 / 0.282).

### Recall by object size

Share of objects found with the right class, by longest side in 320x320 model-input
pixels. "max" = at conf 0.001 (the most a model finds at all); "op" = at its best-F1
threshold.

| size (px) | 0-16 | 16-32 | 32-64 | 64-128 | >=128 |
|---|---|---|---|---|---|
| objects in val | 209 | 271 | 267 | 139 | 17 |
| own, max | **0.450** | **0.808** | **0.880** | **0.914** | **0.941** |
| own, op | 0.134 | 0.487 | 0.655 | 0.712 | 0.706 |
| YOLOv8n default, max | 0.316 | 0.723 | 0.839 | 0.871 | 0.765 |
| YOLOv8n default, op | 0.124 | 0.494 | 0.644 | 0.719 | 0.647 |
| YOLOv8n no-flip + rfs, max | 0.321 | 0.697 | 0.843 | 0.871 | 0.882 |
| YOLOv8n no-flip + rfs, op | 0.105 | 0.461 | 0.618 | 0.734 | 0.765 |

This counts only correctly *classified* objects, so YOLOv8n's weaker classification
lowers it too.

### Per class: own vs. YOLOv8n default

Mean AP@0.5 by how many training objects a class has:

| training objects per class | classes | own | YOLOv8n default |
|---|---|---|---|
| < 20 | 18 | **0.111** | 0.012 |
| 20-99 | 18 | **0.380** | 0.240 |
| 100-499 | 17 | **0.693** | 0.572 |
| >= 500 | 1 | 0.771 | 0.744 |

The gap shrinks as classes get more common and is near zero for the five most common
classes (-0.02 to +0.03). The largest gaps in the common bands were on direction-dependent signs:

| class | own | YOLOv8n default |
|---|---|---|
| `mand_left` | 0.750 | 0.000 |
| `mand_straight_right` | 0.618 | 0.167 |
| `mand_straigh_left` | 0.600 | 0.200 |
| `forb_right` | 0.751 | 0.414 |
| `forb_left` | 0.619 | 0.350 |
| `mand_pass_right` | 0.894 | 0.708 |

Both patterns pointed at the default recipe's mirroring and lack of class rebalancing,
which is what the second YOLOv8n run tested.

## What it shows

1. **The two models have complementary strengths.** YOLOv8n finds and boxes objects
   better (class-agnostic mAP 0.76-0.77 vs 0.68) -- consistent with its stronger box
   loss (Distribution Focal Loss + CIoU vs. plain L1) and its dynamic target
   assignment. This repo's model classifies far better (0.28 vs 0.48 mAP lost to wrong
   classes) and wins overall at 320px: **0.402 vs 0.284 mAP@0.5**.
2. **Mirroring and class balance are not the explanation.** Removing the flips and
   adding repeat-factor sampling left YOLOv8n's mAP unchanged (0.283 -> 0.284) and its
   classification nearly so (0.486 -> 0.475 lost to wrong classes). It made YOLOv8n
   more precise at its operating point (0.69 -> 0.80), but not a better classifier.
3. **Likely causes, untested** (no further runs were budgeted):
   - **Mosaic at 320px.** Mosaic packs four images into one, roughly halving each: a
     median 29px sign is trained at ~15px most of the time -- enough to find a sign,
     often not enough to read "30" vs "50" or an arrow's direction. The own model never
     shrinks images below 0.7x. Fits the pattern exactly: good localization, poor
     classification.
   - **Loss weighting.** Ultralytics weights its box losses well above the class loss
     (box 7.5, DFL 1.5, class 0.5).
4. **The own model is very confident:** its best-F1 threshold is 0.97, its scores
   bunch near 1 -- consistent with the late overfitting in run04.

## Caveats

- val split only, and the own model's recipe was tuned on it; test numbers pending.
- One training run per model: run-to-run noise is unknown (the own model's checkpoints
  move by about +-0.01 near the end of training).
- Ultralytics picks its `best.pt` with its own score (weighted 90% to mAP@[.5:.95]),
  the own model by mAP@0.5.
- Resizing differs (letterbox vs stretch); not isolated.

## Next

- YOLOv8n fine-tuned from COCO weights (no flips, repeat-factor sampling)
- Final three-way numbers on the test split
- For the own model, the localization gap is the improvement list: box loss
  (DFL + CIoU), task-aligned target assignment, separate box/class head branches
