import torch
import torchvision.transforms as T
from PIL import Image

FILL = (114, 114, 114)  # Ultralytics' letterbox grey -- neutral, not a colour any sign uses


def scale_translate(image, boxes, labels, scale, offset, min_visible=0.4, min_size=2.0):
    """Deterministic core of the geometric augmentation. image: a square PIL image
    (already resized to img_size); boxes: (N, 4) xyxy pixels in that image.

    The image is resized by `scale` and pasted onto a same-size grey canvas with its
    top-left corner at `offset` (dx, dy) -- scale < 1 zooms out (grey border), scale > 1
    zooms in (edges cropped). Boxes follow the same transform, then are clipped to the
    canvas. A box is dropped if less than `min_visible` of its area survives the crop
    (training on a sliver of a sign teaches the wrong thing) or if either side ends up
    under `min_size` pixels.
    """
    size = image.size[0]
    dx, dy = offset

    scaled_size = max(1, round(size * scale))
    canvas = Image.new("RGB", (size, size), FILL)
    canvas.paste(image.resize((scaled_size, scaled_size)), (round(dx), round(dy)))

    if boxes.shape[0] == 0:
        return canvas, boxes, labels

    moved = boxes * scale
    moved[:, [0, 2]] += dx
    moved[:, [1, 3]] += dy
    clipped = moved.clamp(min=0, max=size)

    def area(b):
        return (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)

    visible = area(clipped) / area(moved).clamp(min=1e-9)
    keep = (
        (visible >= min_visible)
        & (clipped[:, 2] - clipped[:, 0] >= min_size)
        & (clipped[:, 3] - clipped[:, 1] >= min_size)
    )
    return canvas, clipped[keep], labels[keep]


def random_scale_translate(image, boxes, labels, scale_range=(0.7, 1.3), translate=0.1):
    """Random scale in scale_range, then a random shift of up to `translate` x image
    size in each direction, around the centred position. No rotation and no flips:
    a horizontally flipped "turn left" sign is a "turn right" sign with the wrong label.
    """
    size = image.size[0]
    lo, hi = scale_range
    scale = lo + (hi - lo) * torch.rand(1).item()

    centred = (size - size * scale) / 2
    shift = (torch.rand(2) * 2 - 1) * translate * size
    offset = (centred + shift[0].item(), centred + shift[1].item())

    return scale_translate(image, boxes, labels, scale, offset)


# hue kept small on purpose: colour carries class meaning here (red prohibition vs.
# blue mandatory vs. yellow warning signs), so a large hue shift would relabel them
color_jitter = T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.015)
