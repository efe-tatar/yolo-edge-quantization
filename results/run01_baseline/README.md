# run01_baseline

First successful end-to-end training run of this repo's own model, after the three
fixes described in [../debugging/README.md](../debugging/README.md).
Commit: `7717049`.

## Configuration

| | |
|---|---|
| input size | 320 x 320 (stretched, not letterboxed) |
| strides / size thresholds | 8, 16, 32 / 32px, 80px (from this dataset's box-size percentiles) |
| epochs / batch size | 50 / 16 |
| optimizer | Adam, lr 1e-3, constant (no warmup, no decay, no weight decay) |
| loss | box_weight 5.0, `cls_norm="positive"`, `obj_norm="balanced"` (obj 1.0 / noobj 0.5) |
| sampling | repeat-factor sampling, threshold 0.01 |
| augmentation | none |
| eval | val split, conf_thresh 0.001, per-class NMS IoU 0.5, own mAP implementation |
| hardware | Colab GPU, resumed across sessions from `latest.pt` |

## Results

| metric | value |
|---|---|
| **best val mAP@0.5** | **0.1559** (epoch 46 -- `best.pt`) |
| final val mAP@0.5 | 0.1270 (epoch 49) |
| val mAP@0.5 plateau | ~0.13-0.15 from epoch ~28 onward |
| lowest val cls loss | 8.36 (epoch 18) -- rose to 11-13 afterwards |
| final train cls loss | 0.59 (as low as 0.23 at epoch 37) |

Full per-epoch log: [train_log.txt](train_log.txt).

For reference (not directly comparable -- different split, different mAP
implementation, different training setup): a COCO-pretrained YOLOv8n fine-tuned on
the same dataset in an earlier project reached mAP@0.5 = 0.79.

## Predictions on test images

`best.pt`, conf_thresh 0.3, per-class NMS. Green = ground truth, red = predictions.

![test predictions](test_predictions.png)

## Observations

1. **Classification overfits; localization doesn't.** Val cls loss bottoms out at
   epoch 18 and climbs back while train cls loss keeps falling by ~10x. Val box loss
   keeps improving the whole run. Visually: most red boxes sit on real signs, often
   with the wrong class.
2. **Several boxes per sign, with different labels.** Cells next to an object's
   assigned cell also fire, each with its own argmax class. Per-class NMS only
   suppresses within a class, so all of them survive.
3. **Large stray boxes** (bottom-left: the box over the grass; top-right: the box
   over the barriers). The box head is only ever trained at the single assigned cell
   per object, so when a neighbouring cell's objectness fires, its box output is
   untrained.
4. **Loss spikes at epochs 29 and 48** (train cls 0.50 -> 2.24 at epoch 29):
   consistent with a constant 1e-3 learning rate and no decay.

## Next

- Augmentation (colour jitter, box-aware scale/translate; **no** horizontal flips,
  they swap left/right signs) -> observation 1
- LR warmup + cosine decay, AdamW weight decay -> observations 1, 4
- Center sampling (several positive cells per object) -> observations 2, 3
- Class-agnostic NMS for visualization/deployment -> observation 2
