"""CLI entrypoint for training. The repo stays dataset-agnostic -- this script only
ever touches a manifest.json (data/coco_split.build_manifest's output), never
Roboflow or any dataset-specific detail directly.

Named train_model.py, not train.py -- the latter collides with the train/ package
(import train would resolve to the package, not this file).

Usage:
    python train_model.py --manifest manifest.json --out-dir runs/own_model --num-classes 54
    python train_model.py --manifest manifest.json --out-dir runs/own_model --num-classes 54  # reruns resume automatically
"""
import argparse
import json

import torch
from torch.utils.data import DataLoader

from data.dataset import DetectionDataset, collate_fn
from models.yolo import YOLO
from train.sampler import repeat_factors, RepeatFactorSampler
from train.schedule import param_groups, warmup_cosine
from train.trainer import train


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help="path to a manifest.json from build_manifest")
    parser.add_argument("--out-dir", required=True, help="checkpoints (latest.pt/best.pt) go here")
    parser.add_argument("--num-classes", type=int, required=True)

    parser.add_argument("--img-size", type=int, default=320)
    parser.add_argument("--strides", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--thresholds", type=float, nargs="+", default=[64.0, 128.0],
                         help="placeholder -- override after checking this dataset's real "
                              "box-size distribution (see colab/dataset_split.ipynb)")

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3, help="peak lr, reached after warmup")
    parser.add_argument("--warmup-epochs", type=float, default=3.0)
    parser.add_argument("--final-lr-ratio", type=float, default=0.01,
                         help="cosine decay ends at lr * this, at the last epoch")
    parser.add_argument("--weight-decay", type=float, default=0.01,
                         help="AdamW decoupled weight decay, conv/linear weights only")
    parser.add_argument("--box-weight", type=float, default=5.0)
    parser.add_argument("--center-radius", type=int, default=1,
                         help="center sampling: cells within this many of an object's center "
                              "cell (and inside its box) are also positives; 0 = center cell only")

    parser.add_argument("--no-augment", action="store_true",
                         help="disable training augmentation (scale/translate + colour jitter)")
    parser.add_argument("--aug-scale", type=float, nargs=2, default=[0.7, 1.3])
    parser.add_argument("--aug-translate", type=float, default=0.1,
                         help="max shift, as a fraction of image size")
    parser.add_argument("--num-workers", type=int, default=2,
                         help="DataLoader worker processes (image decode + augmentation)")

    parser.add_argument("--repeat-threshold", type=float, default=0.01,
                         help="LVIS Repeat Factor Sampling threshold -- classes at or above "
                              "this image-frequency are never upweighted")
    parser.add_argument("--no-repeat-sampler", action="store_true",
                         help="use plain shuffling instead of repeat-factor sampling")

    parser.add_argument("--conf-thresh", type=float, default=0.001,
                         help="validation mAP's confidence cutoff -- keep it low, mAP already "
                              "sweeps over confidence (not a deployment threshold)")
    parser.add_argument("--iou-thresh", type=float, default=0.5, help="NMS IoU threshold")
    parser.add_argument("--map-iou-thresholds", type=float, nargs="+", default=[0.5])
    parser.add_argument("--map-every", type=int, default=5,
                         help="compute val mAP every N epochs (always on the last one); "
                              "best.pt is only updated on those epochs")

    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def build_dataloaders(manifest, args):
    train_ds = DetectionDataset(
        manifest["train"], img_size=args.img_size, augment=not args.no_augment,
        scale_range=tuple(args.aug_scale), translate=args.aug_translate,
    )
    loader_kwargs = dict(batch_size=args.batch_size, collate_fn=collate_fn,
                         num_workers=args.num_workers, persistent_workers=args.num_workers > 0)

    if args.no_repeat_sampler:
        train_sampler = None
        train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    else:
        labels_per_image = [sample["labels"] for sample in manifest["train"]]
        factors = repeat_factors(labels_per_image, args.num_classes, args.repeat_threshold)
        train_sampler = RepeatFactorSampler(factors, seed=args.seed)
        train_loader = DataLoader(train_ds, sampler=train_sampler, **loader_kwargs)

    val_loader = None
    if manifest.get("valid"):
        val_ds = DetectionDataset(manifest["valid"], img_size=args.img_size)  # never augmented
        val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)

    return train_loader, val_loader, train_sampler


def main(args):
    torch.manual_seed(args.seed)

    with open(args.manifest) as f:
        manifest = json.load(f)

    train_loader, val_loader, train_sampler = build_dataloaders(manifest, args)

    model = YOLO(args.num_classes).to(args.device)
    optimizer = torch.optim.AdamW(param_groups(model, args.weight_decay), lr=args.lr)
    lr_schedule = warmup_cosine(args.lr, args.epochs, args.warmup_epochs, args.final_lr_ratio)

    train(
        model, train_loader, optimizer, args.device, args.epochs, args.img_size,
        tuple(args.strides), args.num_classes, tuple(args.thresholds), args.out_dir,
        val_loader=val_loader, box_weight=args.box_weight, train_sampler=train_sampler,
        map_iou_thresholds=tuple(args.map_iou_thresholds),
        conf_thresh=args.conf_thresh, iou_thresh=args.iou_thresh, lr_schedule=lr_schedule,
        map_every=args.map_every, center_radius=args.center_radius,
    )


if __name__ == "__main__":
    main(parse_args())
