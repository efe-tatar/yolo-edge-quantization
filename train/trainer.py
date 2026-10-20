import os

import torch

from train.targets import build_targets_batch
from train.loss import yolo_loss


def train_one_epoch(model, dataloader, optimizer, device, img_size, strides, num_classes,
                     thresholds, box_weight=5.0):
    """One pass over dataloader. Yields (images, boxes_list, labels_list) per batch,
    where boxes_list/labels_list are length-B lists of per-image (ragged) tensors --
    the same contract as data/dataset.py's collate_fn, whenever that comes back."""
    model.train()
    totals = {"total": 0.0, "box": 0.0, "cls": 0.0}
    n_batches = 0

    for images, boxes_list, labels_list in dataloader:
        images = images.to(device)
        boxes_list = [b.to(device) for b in boxes_list]
        labels_list = [l.to(device) for l in labels_list]

        preds = model(images)

        targets = build_targets_batch(
            boxes_list, labels_list, img_size, img_size, strides, num_classes, thresholds
        )
        targets = [(bt.to(device), ct.to(device)) for bt, ct in targets]

        losses = yolo_loss(preds, targets, box_weight=box_weight)

        optimizer.zero_grad()
        losses["total"].backward()
        optimizer.step()

        totals["total"] += losses["total"].item()
        totals["box"] += losses["box"].item()
        totals["cls"] += losses["cls"].item()
        n_batches += 1

    return {k: v / n_batches for k, v in totals.items()}


@torch.no_grad()
def evaluate_loss(model, dataloader, device, img_size, strides, num_classes, thresholds,
                   box_weight=5.0):
    """Same loop as train_one_epoch but no gradient step -- for a validation split."""
    model.eval()
    totals = {"total": 0.0, "box": 0.0, "cls": 0.0}
    n_batches = 0

    for images, boxes_list, labels_list in dataloader:
        images = images.to(device)
        boxes_list = [b.to(device) for b in boxes_list]
        labels_list = [l.to(device) for l in labels_list]

        preds = model(images)
        targets = build_targets_batch(
            boxes_list, labels_list, img_size, img_size, strides, num_classes, thresholds
        )
        targets = [(bt.to(device), ct.to(device)) for bt, ct in targets]

        losses = yolo_loss(preds, targets, box_weight=box_weight)

        totals["total"] += losses["total"].item()
        totals["box"] += losses["box"].item()
        totals["cls"] += losses["cls"].item()
        n_batches += 1

    return {k: v / n_batches for k, v in totals.items()}


def save_checkpoint(path, model, optimizer, epoch):
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        },
        path,
    )


def load_checkpoint(path, model, optimizer, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    return ckpt["epoch"]


def train(model, train_loader, optimizer, device, epochs, img_size, strides, num_classes,
          thresholds, checkpoint_path, val_loader=None, box_weight=5.0, log_every=1):
    """Resumes automatically if checkpoint_path already exists -- Colab sessions drop,
    so every epoch is saved, and the next run picks up right after the last completed one."""
    model.to(device)
    start_epoch = 0

    if os.path.exists(checkpoint_path):
        start_epoch = load_checkpoint(checkpoint_path, model, optimizer, device) + 1
        print(f"resumed from checkpoint, starting at epoch {start_epoch}")

    for epoch in range(start_epoch, epochs):
        train_stats = train_one_epoch(
            model, train_loader, optimizer, device, img_size, strides, num_classes,
            thresholds, box_weight,
        )

        msg = f"epoch {epoch}: train total={train_stats['total']:.4f} box={train_stats['box']:.4f} cls={train_stats['cls']:.4f}"

        if val_loader is not None:
            val_stats = evaluate_loss(
                model, val_loader, device, img_size, strides, num_classes, thresholds, box_weight
            )
            msg += f" | val total={val_stats['total']:.4f} box={val_stats['box']:.4f} cls={val_stats['cls']:.4f}"

        if epoch % log_every == 0:
            print(msg)

        save_checkpoint(checkpoint_path, model, optimizer, epoch)

    return model
