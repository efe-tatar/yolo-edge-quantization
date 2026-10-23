import torch


def assign_level(size, thresholds):
    """size: max(box_w, box_h). thresholds: len(strides)-1 size cutoffs, e.g. (64, 128)
    for strides (8, 16, 32) -- upper bound is inclusive at each level."""
    for level, t in enumerate(thresholds):
        if size <= t:
            return level
    return len(thresholds)


def build_targets(boxes, labels, img_h, img_w, strides, num_classes, thresholds, center_radius=0):
    """Build per-level targets for a single image, multi-scale, multi-class.

    boxes: Tensor (N, 4) in absolute pixel coords, (x1, y1, x2, y2).
    labels: Tensor (N,) of class indices, one per box.
    center_radius: FCOS-style center sampling. 0 = only the cell containing the box
        center is positive (the original assignment). r > 0 = also every cell within
        r cells of it (a (2r+1)^2 window) whose center lies inside the box -- so its
        (l, t, r, b) targets are all positive, i.e. a valid box seen from that cell.
        Why (run01/run02): with one positive cell per object, the neighbouring cells
        were trained as background yet still fired at inference, each with its own
        class guess and an untrained box, producing stacks of duplicate boxes that
        per-class NMS can't merge. Training them as positives gives them the right
        class and a real box, so NMS collapses them -- and gives the classifier
        several examples per object instead of one. Tiny boxes may contain no
        neighbouring cell centers and keep just the center cell, as before.

    Returns a list of (box_target, obj_target, cls_target) triples, one per level:
        box_target: (4, H, W) -- (l, t, r, b) distances from the cell center to the
                    box edges, in units of that level's stride (not raw pixels).
        obj_target: (1, H, W) -- 1.0 at the assigned cell, 0.0 elsewhere. Decoupled
                    from cls_target so classification loss can be gated to positive
                    cells only (see train/loss.py) -- the YOLOv1-style split, rather
                    than merging "is there an object" into the class scores.
        cls_target: (num_classes, H, W) -- one-hot at the assigned cell, meaningless
                    (and unused) elsewhere, since cls loss never looks at those cells.

    On a collision (two boxes assigned the same cell), the smaller-area box wins --
    the standard FCOS tie-break, prioritizing the harder, smaller object. Decided per
    cell, so with center sampling two nearby boxes can each keep part of their window.
    """
    grid_sizes = [(img_h // s, img_w // s) for s in strides]
    targets = [
        (torch.zeros(4, h, w), torch.zeros(1, h, w), torch.zeros(num_classes, h, w))
        for h, w in grid_sizes
    ]
    claimed_area = [torch.full((h, w), float("inf")) for h, w in grid_sizes]

    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = box.tolist()
        w_box, h_box = x2 - x1, y2 - y1
        size = max(w_box, h_box)
        area = w_box * h_box

        level = assign_level(size, thresholds)
        stride = strides[level]
        box_target, obj_target, cls_target = targets[level]
        h, w = grid_sizes[level]

        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        center_x = min(max(int(cx // stride), 0), w - 1)
        center_y = min(max(int(cy // stride), 0), h - 1)

        for dy in range(-center_radius, center_radius + 1):
            for dx in range(-center_radius, center_radius + 1):
                grid_x, grid_y = center_x + dx, center_y + dy
                if not (0 <= grid_x < w and 0 <= grid_y < h):
                    continue

                cell_cx = (grid_x + 0.5) * stride
                cell_cy = (grid_y + 0.5) * stride

                # the center cell is always kept (even if a tiny box doesn't contain
                # its center -- the original behaviour); neighbours only when inside
                is_center = dx == 0 and dy == 0
                if not is_center and not (x1 < cell_cx < x2 and y1 < cell_cy < y2):
                    continue

                if area >= claimed_area[level][grid_y, grid_x]:
                    continue  # a smaller box already claimed this cell
                claimed_area[level][grid_y, grid_x] = area

                l = (cell_cx - x1) / stride
                t = (cell_cy - y1) / stride
                r = (x2 - cell_cx) / stride
                b = (y2 - cell_cy) / stride

                box_target[:, grid_y, grid_x] = torch.tensor([l, t, r, b])
                obj_target[0, grid_y, grid_x] = 1.0
                cls_target[:, grid_y, grid_x] = 0.0  # clear in case a since-replaced box set it
                cls_target[int(label), grid_y, grid_x] = 1.0

    return targets


def build_targets_batch(boxes_list, labels_list, img_h, img_w, strides, num_classes, thresholds,
                        center_radius=0):
    """boxes_list/labels_list: length B, per-image GT (ragged counts).

    Returns a list of (box_target, obj_target, cls_target) triples, one per level --
    same structure as build_targets, but each tensor now has a batch dimension.
    """
    per_image = [
        build_targets(boxes, labels, img_h, img_w, strides, num_classes, thresholds, center_radius)
        for boxes, labels in zip(boxes_list, labels_list)
    ]

    batched = []
    for level in range(len(strides)):
        box_targets = torch.stack([triple[level][0] for triple in per_image])
        obj_targets = torch.stack([triple[level][1] for triple in per_image])
        cls_targets = torch.stack([triple[level][2] for triple in per_image])
        batched.append((box_targets, obj_targets, cls_targets))
    return batched


def decode_boxes(box_pred, stride):
    """Turn raw (l, t, r, b) predictions (in units of stride) back into absolute
    (x1, y1, x2, y2) pixel boxes.

    box_pred: (B, 4, H, W). Returns (B, H, W, 4).
    """
    b, _, h, w = box_pred.shape
    device = box_pred.device

    ys = (torch.arange(h, device=device) + 0.5) * stride
    xs = (torch.arange(w, device=device) + 0.5) * stride
    cell_cy, cell_cx = torch.meshgrid(ys, xs, indexing="ij")

    l, t, r, bo = box_pred.unbind(dim=1)
    # targets are always >=0 by construction (a cell's center is always inside its
    # box), but the raw network output isn't constrained that way -- an undertrained
    # model can predict a negative distance, which would otherwise decode to an
    # inverted box (x1 > x2), invalid for anything downstream (PIL, IoU, NMS)
    l, t, r, bo = l.clamp(min=0), t.clamp(min=0), r.clamp(min=0), bo.clamp(min=0)
    l, t, r, bo = l * stride, t * stride, r * stride, bo * stride

    x1 = cell_cx.unsqueeze(0) - l
    y1 = cell_cy.unsqueeze(0) - t
    x2 = cell_cx.unsqueeze(0) + r
    y2 = cell_cy.unsqueeze(0) + bo

    return torch.stack([x1, y1, x2, y2], dim=-1)
