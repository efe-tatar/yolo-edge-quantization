"""Export a manifest (data/coco_split.build_manifest's output) to the folder layout
Ultralytics trains on, so the Ultralytics baselines see exactly the same split as
this repo's own model:

    out_dir/
        images/{train,val,test}/<name>.jpg
        labels/{train,val,test}/<name>.txt   one "class cx cy w h" line per box,
                                             all four normalized to 0-1 by image size
        data.yaml

Ultralytics finds an image's label file by swapping /images/ for /labels/ in its
path, so the two trees must mirror each other exactly.
"""
import json
import os
import random
import shutil

from PIL import Image

# manifest split name -> Ultralytics split name
SPLITS = {"train": "train", "valid": "val", "test": "test"}


def to_yolo_lines(boxes, labels, img_w, img_h):
    """boxes: [[x1, y1, x2, y2], ...] in absolute pixels of the ORIGINAL image (the
    manifest's convention). Returns "class cx cy w h" lines, normalized to 0-1.

    Boxes are clipped to the image first: a box poking past the border would give
    a normalized coordinate outside 0-1, which Ultralytics drops with a warning --
    silently losing a label. Boxes with no area left after clipping are skipped.
    """
    lines = []
    for (x1, y1, x2, y2), label in zip(boxes, labels):
        x1, x2 = max(0.0, min(x1, img_w)), max(0.0, min(x2, img_w))
        y1, y2 = max(0.0, min(y1, img_h)), max(0.0, min(y2, img_h))
        w, h = x2 - x1, y2 - y1
        if w <= 0 or h <= 0:
            continue
        cx, cy = x1 + w / 2, y1 + h / 2
        lines.append(f"{int(label)} {cx / img_w:.6f} {cy / img_h:.6f} {w / img_w:.6f} {h / img_h:.6f}")
    return lines


def copies_per_image(repeat_factors, seed=42):
    """Turns repeat factors (train/sampler.repeat_factors) into a fixed number of
    copies per image: floor(r), plus one more with probability frac(r). The expected
    count per image equals r -- same as RepeatFactorSampler, except that one draws
    again every epoch while these copies are drawn once, at dataset build time
    (Ultralytics has no per-epoch sampler hook). Seeded, so a rebuilt dataset (e.g.
    for a resumed training) is identical."""
    rng = random.Random(seed)
    counts = []
    for r in repeat_factors:
        whole = int(r)
        counts.append(whole + (1 if rng.random() < r - whole else 0))
    return counts


def write_yolo_dataset(manifest, class_names, out_dir, train_repeat_factors=None, seed=42):
    """Copies every manifest image into out_dir and writes its label file, plus
    data.yaml. Returns the path to data.yaml.

    Images are copied, not symlinked: the copies live on the Colab VM's local disk
    (fast reads during training), and nothing depends on how Ultralytics resolves
    links. Each file gets its manifest index as a prefix -- the manifest pools
    Roboflow's original train/valid/test folders, so two images could share a name.

    train_repeat_factors: optional, one factor per manifest["train"] sample --
    repeat-factor sampling for Ultralytics, by writing extra copies of train images
    (see copies_per_image). val and test are never repeated.
    """
    out_dir = os.path.abspath(out_dir)

    for manifest_split, yolo_split in SPLITS.items():
        image_dir = os.path.join(out_dir, "images", yolo_split)
        label_dir = os.path.join(out_dir, "labels", yolo_split)
        os.makedirs(image_dir, exist_ok=True)
        os.makedirs(label_dir, exist_ok=True)

        samples = manifest.get(manifest_split, [])
        if manifest_split == "train" and train_repeat_factors is not None:
            assert len(train_repeat_factors) == len(samples), "one repeat factor per train image"
            counts = copies_per_image(train_repeat_factors, seed)
        else:
            counts = [1] * len(samples)

        for i, (sample, count) in enumerate(zip(samples, counts)):
            src = sample["image_path"]
            with Image.open(src) as image:  # reads the header only, not the pixels
                img_w, img_h = image.size
            lines = to_yolo_lines(sample["boxes"], sample["labels"], img_w, img_h)

            for k in range(count):
                copy_tag = f"_rep{k}" if k else ""
                stem, ext = os.path.splitext(f"{i:06d}{copy_tag}_{os.path.basename(src)}")
                shutil.copyfile(src, os.path.join(image_dir, stem + ext))
                with open(os.path.join(label_dir, stem + ".txt"), "w") as f:
                    f.write("\n".join(lines) + ("\n" if lines else ""))

    # written by hand rather than with a YAML library: a JSON string is valid YAML,
    # so json.dumps quotes every class name safely
    yaml_lines = [
        f"path: {json.dumps(out_dir)}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "names:",
    ] + [f"  {i}: {json.dumps(name)}" for i, name in enumerate(class_names)]

    data_yaml = os.path.join(out_dir, "data.yaml")
    with open(data_yaml, "w") as f:
        f.write("\n".join(yaml_lines) + "\n")
    return data_yaml
