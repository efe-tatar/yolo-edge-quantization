import os

from data.coco_split import pool_boxes, class_photo_counts, redistribute, build_manifest


def make_coco(images_and_boxes, categories):
    """images_and_boxes: list of (image_id, file_name, [(category_id, bbox), ...])."""
    images = []
    annotations = []
    ann_id = 0
    for image_id, file_name, boxes in images_and_boxes:
        images.append({"id": image_id, "file_name": file_name})
        for category_id, bbox in boxes:
            annotations.append({
                "id": ann_id, "image_id": image_id, "category_id": category_id, "bbox": bbox,
            })
            ann_id += 1
    return {"images": images, "annotations": annotations, "categories": categories}


def test_pool_boxes_flattens_across_splits():
    categories = [{"id": 0, "name": "cat"}]
    coco_by_split = {
        "train": make_coco([(1, "a.jpg", [(0, [0, 0, 10, 10])])], categories),
        "valid": make_coco([(2, "b.jpg", [(0, [1, 1, 5, 5])])], categories),
    }

    boxes = pool_boxes(coco_by_split)

    assert len(boxes) == 2
    train_box = next(b for b in boxes if b["split"] == "train")
    assert train_box == {
        "split": "train", "imageId": 1, "file": "a.jpg", "class": "cat",
        "x": 0, "y": 0, "width": 10, "height": 10,
    }


def test_pool_boxes_drops_degenerate_boxes():
    categories = [{"id": 0, "name": "cat"}]
    coco_by_split = {
        "train": make_coco([(1, "a.jpg", [(0, [0, 0, 0.5, 0.5]), (0, [0, 0, 10, 10])])], categories),
    }

    boxes = pool_boxes(coco_by_split)
    assert len(boxes) == 1  # the 0.5x0.5 box was dropped


def test_class_photo_counts_counts_distinct_photos_not_boxes():
    categories = [{"id": 0, "name": "cat"}]
    # two boxes of the same class on the SAME photo -> 2 boxes, 1 photo
    coco_by_split = {
        "train": make_coco([(1, "a.jpg", [(0, [0, 0, 5, 5]), (0, [10, 10, 15, 15])])], categories),
    }
    boxes = pool_boxes(coco_by_split)

    class_names, boxes_per_class, images_per_class = class_photo_counts(boxes)
    assert boxes_per_class["cat"] == 2
    assert len(images_per_class["cat"]) == 1


def _boxes_for_class(class_name, photo_ids, split="s"):
    return [
        {"split": split, "imageId": i, "file": f"{i}.jpg", "class": class_name,
         "x": 0, "y": 0, "width": 10, "height": 10}
        for i in photo_ids
    ]


def test_redistribute_splits_a_well_represented_class_roughly_70_15_15():
    boxes = _boxes_for_class("common", range(10))

    set_of_photo, train_only = redistribute(boxes, min_photos=4, seed=42)

    assert train_only == []
    counts = {"train": 0, "valid": 0, "test": 0}
    for i in range(10):
        counts[set_of_photo[("s", i)]] += 1
    assert counts == {"train": 7, "valid": 1, "test": 2}  # int(0.7*10)=7, int(0.85*10)=8


def test_redistribute_forces_rare_class_train_only():
    boxes = _boxes_for_class("rare", [100, 101])  # only 2 distinct photos, below min_photos=4

    set_of_photo, train_only = redistribute(boxes, min_photos=4, seed=42)

    assert train_only == ["rare"]
    assert set_of_photo[("s", 100)] == "train"
    assert set_of_photo[("s", 101)] == "train"


def test_redistribute_is_independent_of_input_box_order():
    # same boxes, two different list orders -- images_per_class is built from a set
    # internally, whose iteration order is NOT guaranteed stable across Python
    # processes (string hash randomization), so this must not leak into the result
    boxes_forward = _boxes_for_class("common", range(10)) + _boxes_for_class("rare", [100, 101])
    boxes_reversed = list(reversed(boxes_forward))

    result_forward = redistribute(boxes_forward, min_photos=4, seed=42)
    result_reversed = redistribute(boxes_reversed, min_photos=4, seed=42)

    assert result_forward == result_reversed


def test_redistribute_locks_a_shared_photo_to_the_rarer_classs_choice():
    # "rare" (2 boxes) is processed before "common" (4 boxes), and they share photo p1
    boxes = (
        _boxes_for_class("common", [1, 2, 3, 4])
        + _boxes_for_class("rare", [1, 5])
    )

    set_of_photo, train_only = redistribute(boxes, min_photos=4, seed=42)

    assert train_only == ["rare"]
    # rare claimed photo 1 as train-only; common must respect that, not overwrite it
    assert set_of_photo[("s", 1)] == "train"
    # every photo ends up assigned, none left over
    for photo_id in [1, 2, 3, 4, 5]:
        assert ("s", photo_id) in set_of_photo


def test_build_manifest_groups_by_photo_and_converts_boxes():
    boxes = [
        {"split": "train", "imageId": 1, "file": "a.jpg", "class": "cat",
         "x": 10, "y": 20, "width": 5, "height": 6},
        {"split": "train", "imageId": 1, "file": "a.jpg", "class": "dog",
         "x": 0, "y": 0, "width": 2, "height": 3},
        {"split": "train", "imageId": 2, "file": "b.jpg", "class": "cat",
         "x": 1, "y": 1, "width": 4, "height": 4},
    ]
    set_of_photo = {("train", 1): "train", ("train", 2): "valid"}
    class_to_index = {"cat": 0, "dog": 1}

    manifest = build_manifest(boxes, set_of_photo, class_to_index, "/data")

    assert len(manifest["train"]) == 1
    assert len(manifest["valid"]) == 1
    assert manifest["test"] == []

    photo1 = manifest["train"][0]
    assert photo1["image_path"] == os.path.join("/data", "train", "a.jpg")
    assert photo1["boxes"] == [[10, 20, 15, 26], [0, 0, 2, 3]]
    assert photo1["labels"] == [0, 1]

    photo2 = manifest["valid"][0]
    assert photo2["boxes"] == [[1, 1, 5, 5]]
    assert photo2["labels"] == [0]
