from PIL import ImageDraw


def draw_detections(image, boxes, labels, scores=None, class_names=None, color="red"):
    """Draws boxes + labels on a COPY of image (the input is never modified).

    boxes: (N, 4) xyxy, in the same coordinate frame as `image` (not rescaled here).
    labels: (N,) int class indices.
    scores: optional (N,) confidences -- omit for ground truth, which has none.
    class_names: optional, index -> name, for readable labels instead of raw ints.
    color: lets a caller draw predictions and ground truth in different colors on
    the same image for a side-by-side sanity check.
    """
    image = image.copy()
    draw = ImageDraw.Draw(image)

    for i in range(len(boxes)):
        x1, y1, x2, y2 = [float(v) for v in boxes[i]]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

        name = class_names[int(labels[i])] if class_names is not None else str(int(labels[i]))
        text = name if scores is None else f"{name} {float(scores[i]):.2f}"
        draw.text((x1, max(0, y1 - 10)), text, fill=color)

    return image
