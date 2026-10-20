import math
import random

import torch


def category_image_frequencies(labels_per_image, num_classes):
    """labels_per_image: list of per-image label tensors/lists, one entry per
    dataset sample, in index order.

    Returns a list of length num_classes: the FRACTION of images containing at
    least one instance of that class -- an image with 50 boxes of one class still
    counts once, same convention as data.coco_split.class_photo_counts.
    """
    n_images = len(labels_per_image)
    image_count_per_class = [0] * num_classes

    for labels in labels_per_image:
        present_classes = {int(l) for l in labels}
        for c in present_classes:
            image_count_per_class[c] += 1

    return [count / n_images for count in image_count_per_class]


def repeat_factors(labels_per_image, num_classes, repeat_threshold=0.01):
    """LVIS's Repeat Factor Sampling recipe. Returns one repeat factor (float >= 1.0)
    per image -- an image's factor is driven by the rarest class it contains.

    repeat_threshold: classes at or above this image-frequency get factor 1.0
    (never upweighted); rarer classes get sqrt(repeat_threshold / frequency).
    Not a repo default to tune once and forget -- pass the value that matches
    the real dataset's distribution, same principle as assign_level's thresholds.
    """
    class_freq = category_image_frequencies(labels_per_image, num_classes)

    class_repeat = [
        1.0 if freq == 0 else max(1.0, math.sqrt(repeat_threshold / freq))
        for freq in class_freq
    ]

    image_repeats = []
    for labels in labels_per_image:
        if len(labels) == 0:
            image_repeats.append(1.0)  # a background image -- nothing to upweight it by
        else:
            image_repeats.append(max(class_repeat[int(l)] for l in labels))

    return image_repeats


class RepeatFactorSampler(torch.utils.data.Sampler):
    """Each image appears floor(r) times for certain, plus one more with
    probability frac(r), re-drawn every epoch via set_epoch -- so the expected
    number of times an image is seen per epoch equals its repeat factor, without
    ever risking a rare image being skipped entirely by chance in a given epoch.
    """

    def __init__(self, image_repeats, seed=42):
        self.image_repeats = image_repeats
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def _expanded_indices(self):
        rng = random.Random(self.seed + self.epoch)
        indices = []
        for idx, r in enumerate(self.image_repeats):
            whole = int(r)
            frac = r - whole
            count = whole + (1 if rng.random() < frac else 0)
            indices.extend([idx] * count)
        rng.shuffle(indices)
        return indices

    def __iter__(self):
        return iter(self._expanded_indices())

    def __len__(self):
        return len(self._expanded_indices())
