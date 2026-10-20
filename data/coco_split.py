import os
import random


def pool_boxes(coco_by_split):
    """coco_by_split: {"<split name>": <parsed COCO json>, ...} -- already loaded,
    so this function has no file I/O and no knowledge of where the json came from.

    Returns a flat list of box dicts: {"split", "imageId", "file", "class",
    "x", "y", "width", "height"}, pooled across every split given.
    """
    boxes = []
    for split_name, coco in coco_by_split.items():
        category_name = {c["id"]: c["name"] for c in coco["categories"]}
        file_of_image = {img["id"]: img["file_name"] for img in coco["images"]}

        for annotation in coco["annotations"]:
            x, y, width, height = annotation["bbox"]
            if width < 1 or height < 1:
                continue  # degenerate box, same filter as the classification pipeline

            boxes.append({
                "split": split_name,
                "imageId": annotation["image_id"],
                "file": file_of_image[annotation["image_id"]],
                "class": category_name[annotation["category_id"]],
                "x": x, "y": y, "width": width, "height": height,
            })
    return boxes


def class_photo_counts(boxes):
    """Returns (class_names, boxes_per_class, images_per_class).

    images_per_class maps each class to the SET of (split, imageId) it appears in --
    that's the unit redistribute() actually splits on, not raw box count.
    """
    class_names = []
    boxes_per_class = {}
    images_per_class = {}

    for box in boxes:
        class_name = box["class"]
        if class_name not in boxes_per_class:
            class_names.append(class_name)
            boxes_per_class[class_name] = 0
            images_per_class[class_name] = set()
        boxes_per_class[class_name] += 1
        images_per_class[class_name].add((box["split"], box["imageId"]))

    return class_names, boxes_per_class, images_per_class


def redistribute(boxes, min_photos=4, seed=42):
    """Rarest-class-first, photo-level 70/15/15 split, keeping every class.

    A class with fewer than min_photos distinct photos can't guarantee every split
    sees it -- with this exact 70/15/15 (int-truncated) formula, min_photos=4 is the
    smallest count where train/valid/test all get at least one photo. Such a class is
    kept (not dropped) but forced train-only.

    Returns (set_of_photo, train_only_classes):
        set_of_photo: {(split, imageId): "train"|"valid"|"test"}
        train_only_classes: list of class names forced train-only
    """
    class_names, boxes_per_class, images_per_class = class_photo_counts(boxes)
    sorted_classes = sorted(class_names, key=boxes_per_class.get, reverse=True)

    rng = random.Random(seed)  # local instance -- never touches the global random module
    set_of_photo = {}
    train_only_classes = []

    for class_name in reversed(sorted_classes):  # rarest first -- gets priority on shared photos
        total_photos = images_per_class[class_name]
        # sorted(), not a plain list comprehension over the set: set iteration order
        # depends on Python's per-process string hash randomization, so without this,
        # the same seed could still shuffle a differently-ordered input each run
        unclaimed = sorted(photo for photo in total_photos if photo not in set_of_photo)
        rng.shuffle(unclaimed)

        if len(total_photos) < min_photos:
            train_only_classes.append(class_name)
            for photo in unclaimed:
                set_of_photo[photo] = "train"
            continue

        train_end = int(0.70 * len(unclaimed))
        valid_end = int(0.85 * len(unclaimed))
        for i, photo in enumerate(unclaimed):
            if i < train_end:
                set_of_photo[photo] = "train"
            elif i < valid_end:
                set_of_photo[photo] = "valid"
            else:
                set_of_photo[photo] = "test"

    return set_of_photo, train_only_classes


def build_manifest(boxes, set_of_photo, class_to_index, dataset_directory):
    """Groups boxes back by photo and emits the per-sample structure DetectionDataset
    expects: {"train": [...], "valid": [...], "test": [...]}, each entry
    {"image_path": str, "boxes": [[x1,y1,x2,y2], ...], "labels": [int, ...]}.
    """
    boxes_by_photo = {}
    for box in boxes:
        photo = (box["split"], box["imageId"])
        boxes_by_photo.setdefault(photo, []).append(box)

    manifest = {"train": [], "valid": [], "test": []}

    for photo, photo_boxes in boxes_by_photo.items():
        split, _ = photo
        set_name = set_of_photo[photo]

        image_path = os.path.join(dataset_directory, split, photo_boxes[0]["file"])

        box_coords = []
        labels = []
        for box in photo_boxes:
            x1, y1 = box["x"], box["y"]
            box_coords.append([x1, y1, x1 + box["width"], y1 + box["height"]])
            labels.append(class_to_index[box["class"]])

        manifest[set_name].append({
            "image_path": image_path,
            "boxes": box_coords,
            "labels": labels,
        })

    return manifest
