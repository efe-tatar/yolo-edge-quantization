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
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--box-weight", type=float, default=5.0)

    parser.add_argument("--repeat-threshold", type=float, default=0.01,
                         help="LVIS Repeat Factor Sampling threshold -- classes at or above "
                              "this image-frequency are never upweighted")
    parser.add_argument("--no-repeat-sampler", action="store_true",
                         help="use plain shuffling instead of repeat-factor sampling")

    parser.add_argument("--conf-thresh", type=float, default=0.3)
    parser.add_argument("--iou-thresh", type=float, default=0.5)
    parser.add_argument("--map-iou-thresholds", type=float, nargs="+", default=[0.5])

    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def build_dataloaders(manifest, args):
    train_ds = DetectionDataset(manifest["train"], img_size=args.img_size)

    if args.no_repeat_sampler:
        train_sampler = None
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
        )
    else:
        labels_per_image = [sample["labels"] for sample in manifest["train"]]
        factors = repeat_factors(labels_per_image, args.num_classes, args.repeat_threshold)
        train_sampler = RepeatFactorSampler(factors, seed=args.seed)
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, sampler=train_sampler, collate_fn=collate_fn
        )

    val_loader = None
    if manifest.get("valid"):
        val_ds = DetectionDataset(manifest["valid"], img_size=args.img_size)
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
        )

    return train_loader, val_loader, train_sampler


def main(args):
    torch.manual_seed(args.seed)

    with open(args.manifest) as f:
        manifest = json.load(f)

    train_loader, val_loader, train_sampler = build_dataloaders(manifest, args)

    model = YOLO(args.num_classes).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train(
        model, train_loader, optimizer, args.device, args.epochs, args.img_size,
        tuple(args.strides), args.num_classes, tuple(args.thresholds), args.out_dir,
        val_loader=val_loader, box_weight=args.box_weight, train_sampler=train_sampler,
        map_iou_thresholds=tuple(args.map_iou_thresholds),
    )


if __name__ == "__main__":
    main(parse_args())
