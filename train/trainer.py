import os

import torch

from train.targets import build_targets_batch
from train.loss import yolo_loss
from train.decode import decode_predictions
from train.metrics import mean_average_precision


def train_one_epoch(model, dataloader, optimizer, device, img_size, strides, num_classes,
                     thresholds, box_weight=5.0):
    """One pass over dataloader. Yields (images, boxes_list, labels_list) per batch,
    where boxes_list/labels_list are length-B lists of per-image (ragged) tensors --
    the same contract as data/dataset.py's collate_fn."""
    model.train()
    totals = {"total": 0.0, "box": 0.0, "obj": 0.0, "cls": 0.0}
    n_batches = 0

    for images, boxes_list, labels_list in dataloader:
        images = images.to(device)
        boxes_list = [b.to(device) for b in boxes_list]
        labels_list = [l.to(device) for l in labels_list]

        preds = model(images)

        targets = build_targets_batch(
            boxes_list, labels_list, img_size, img_size, strides, num_classes, thresholds
        )
        targets = [(bt.to(device), ot.to(device), ct.to(device)) for bt, ot, ct in targets]

        losses = yolo_loss(preds, targets, box_weight=box_weight)

        optimizer.zero_grad()
        losses["total"].backward()
        optimizer.step()

        totals["total"] += losses["total"].item()
        totals["box"] += losses["box"].item()
        totals["obj"] += losses["obj"].item()
        totals["cls"] += losses["cls"].item()
        n_batches += 1

    return {k: v / n_batches for k, v in totals.items()}


@torch.no_grad()
def evaluate_loss(model, dataloader, device, img_size, strides, num_classes, thresholds,
                   box_weight=5.0):
    """Same loop as train_one_epoch but no gradient step -- for a validation split."""
    model.eval()
    totals = {"total": 0.0, "box": 0.0, "obj": 0.0, "cls": 0.0}
    n_batches = 0

    for images, boxes_list, labels_list in dataloader:
        images = images.to(device)
        boxes_list = [b.to(device) for b in boxes_list]
        labels_list = [l.to(device) for l in labels_list]

        preds = model(images)
        targets = build_targets_batch(
            boxes_list, labels_list, img_size, img_size, strides, num_classes, thresholds
        )
        targets = [(bt.to(device), ot.to(device), ct.to(device)) for bt, ot, ct in targets]

        losses = yolo_loss(preds, targets, box_weight=box_weight)

        totals["total"] += losses["total"].item()
        totals["box"] += losses["box"].item()
        totals["obj"] += losses["obj"].item()
        totals["cls"] += losses["cls"].item()
        n_batches += 1

    return {k: v / n_batches for k, v in totals.items()}


@torch.no_grad()
def evaluate_map(model, dataloader, device, strides, num_classes, conf_thresh=0.001,
                  iou_thresh=0.5, map_iou_thresholds=(0.5,), class_agnostic=False):
    """Runs the model over dataloader, decodes predictions (confidence threshold +
    NMS, same path real inference would use), and scores them against ground truth
    with our own mean_average_precision -- not each framework's own eval, so every
    number in a cross-model comparison comes from identical matching/AP code.

    conf_thresh defaults low on purpose (Ultralytics' val uses 0.001 too): mAP already
    integrates over every confidence level, so a deployment-style 0.3 cutoff here just
    truncates the precision/recall curve -- and early in training can drop every true
    positive, reporting exactly 0.0 for a model that is in fact learning.

    class_agnostic: every prediction and GT is relabelled to one class (NMS too), so
    the score measures "found the object, box in the right place" with classification
    taken out -- the gap to the normal mAP is what classification errors cost."""
    model.eval()
    all_preds = []
    all_gts = []

    for images, boxes_list, labels_list in dataloader:
        images = images.to(device)
        preds = model(images)
        decoded = decode_predictions(preds, strides, conf_thresh=conf_thresh, iou_thresh=iou_thresh,
                                     class_agnostic_nms=class_agnostic)
        # decoded predictions are on `device` (wherever the model ran); ground truth
        # from the dataloader is always on CPU -- box_iou needs both on the same one
        decoded = [(b.to("cpu"), s.to("cpu"), l.to("cpu")) for b, s, l in decoded]

        if class_agnostic:
            decoded = [(b, s, torch.zeros_like(l)) for b, s, l in decoded]
            labels_list = [torch.zeros_like(l) for l in labels_list]

        all_preds.extend(decoded)
        all_gts.extend(zip(boxes_list, labels_list))

    n_scored_classes = 1 if class_agnostic else num_classes
    return mean_average_precision(all_preds, all_gts, n_scored_classes, iou_thresholds=map_iou_thresholds)


def save_checkpoint(path, model, optimizer, epoch, extra=None):
    """Written to a temp file then renamed: os.replace is atomic, so a crash or a
    Colab disconnect mid-save never leaves a half-written checkpoint behind."""
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        **(extra or {}),
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_checkpoint(path, model, optimizer=None, device="cpu"):
    state = torch.load(path, map_location=device)
    model.load_state_dict(state["model"])
    if optimizer is not None and state.get("optimizer") is not None:
        optimizer.load_state_dict(state["optimizer"])
    return state


def train(model, train_loader, optimizer, device, epochs, img_size, strides, num_classes,
          thresholds, out_dir, val_loader=None, box_weight=5.0, log_every=1,
          train_sampler=None, map_iou_thresholds=(0.5,), conf_thresh=0.001, iou_thresh=0.5):
    """Resumes automatically from out_dir/latest.pt if it already exists -- Colab
    sessions drop, so every epoch is saved, not just the end. out_dir/best.pt is
    only (re)written when validation mAP@0.5 improves, and holds just the model --
    latest.pt holds model+optimizer+epoch for exact resume.

    train_sampler: pass the RepeatFactorSampler instance (not just the DataLoader
    wrapping it) so its set_epoch can be called -- PyTorch never does this for you.
    """
    model.to(device)
    os.makedirs(out_dir, exist_ok=True)
    latest_path = os.path.join(out_dir, "latest.pt")
    best_path = os.path.join(out_dir, "best.pt")

    start_epoch = 0
    best_map = -1.0

    if os.path.exists(latest_path):
        state = load_checkpoint(latest_path, model, optimizer, device)
        start_epoch = state["epoch"] + 1
        best_map = state.get("best_map", -1.0)
        print(f"resumed from {latest_path}, starting at epoch {start_epoch}")

    for epoch in range(start_epoch, epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, train_loader, optimizer, device, img_size, strides, num_classes,
            thresholds, box_weight,
        )
        msg = f"epoch {epoch}: train total={train_stats['total']:.4f} box={train_stats['box']:.4f} obj={train_stats['obj']:.4f} cls={train_stats['cls']:.4f}"

        val_map = None
        if val_loader is not None:
            val_stats = evaluate_loss(
                model, val_loader, device, img_size, strides, num_classes, thresholds, box_weight
            )
            msg += f" | val total={val_stats['total']:.4f} box={val_stats['box']:.4f} obj={val_stats['obj']:.4f} cls={val_stats['cls']:.4f}"

            map_results = evaluate_map(
                model, val_loader, device, strides, num_classes, conf_thresh=conf_thresh,
                iou_thresh=iou_thresh, map_iou_thresholds=map_iou_thresholds,
            )
            val_map = map_results[map_iou_thresholds[0]]
            msg += f" | val mAP@{map_iou_thresholds[0]}={val_map:.4f}"

        if epoch % log_every == 0:
            print(msg)

        save_checkpoint(latest_path, model, optimizer, epoch, extra={"best_map": best_map})

        if val_map is not None and val_map > best_map:
            best_map = val_map
            save_checkpoint(best_path, model, optimizer, epoch, extra={"val_map": best_map})

    return model
