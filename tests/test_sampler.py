import pytest
import torch

from train.sampler import category_image_frequencies, repeat_factors, RepeatFactorSampler


def test_category_image_frequencies_counts_images_not_boxes():
    # class 0 appears (possibly multiple times) in 8 of 10 images, class 1 in 2 of 10
    labels_per_image = (
        [torch.tensor([0, 0, 0])] * 8          # multiple class-0 boxes, still 1 image each
        + [torch.tensor([1])] * 2
    )
    freqs = category_image_frequencies(labels_per_image, num_classes=2)
    assert freqs == pytest.approx([0.8, 0.2])


def test_repeat_factors_common_class_gets_one():
    labels_per_image = [torch.tensor([0])] * 8 + [torch.tensor([1])] * 2
    factors = repeat_factors(labels_per_image, num_classes=2, repeat_threshold=0.5)

    # class 0: freq 0.8 -> sqrt(0.5/0.8) < 1 -> clamped to 1.0
    for i in range(8):
        assert factors[i] == pytest.approx(1.0)


def test_repeat_factors_rare_class_gets_upweighted():
    labels_per_image = [torch.tensor([0])] * 8 + [torch.tensor([1])] * 2
    factors = repeat_factors(labels_per_image, num_classes=2, repeat_threshold=0.5)

    # class 1: freq 0.2 -> sqrt(0.5/0.2) = sqrt(2.5)
    expected = 2.5 ** 0.5
    for i in range(8, 10):
        assert factors[i] == pytest.approx(expected)


def test_repeat_factor_driven_by_rarest_class_present():
    # one image contains both the common and the rare class -> its factor is the max
    labels_per_image = [torch.tensor([0])] * 8 + [torch.tensor([1])] * 1 + [torch.tensor([0, 1])]
    factors = repeat_factors(labels_per_image, num_classes=2, repeat_threshold=0.5)

    rare_factor = 2.5 ** 0.5
    assert factors[-1] == pytest.approx(rare_factor)  # driven by class 1, not class 0


def test_background_image_gets_factor_one():
    labels_per_image = [torch.tensor([0])] * 8 + [torch.tensor([1])] * 2 + [torch.tensor([], dtype=torch.long)]
    factors = repeat_factors(labels_per_image, num_classes=2, repeat_threshold=0.5)
    assert factors[-1] == pytest.approx(1.0)


def test_sampler_count_per_epoch_bounded_by_floor_and_ceil():
    sampler = RepeatFactorSampler(image_repeats=[2.5], seed=0)
    for epoch in range(50):
        sampler.set_epoch(epoch)
        indices = list(sampler)
        assert len(indices) in (2, 3)
        assert all(i == 0 for i in indices)


def test_sampler_stochastic_rounding_averages_to_the_repeat_factor():
    sampler = RepeatFactorSampler(image_repeats=[2.5], seed=0)
    counts = []
    for epoch in range(400):
        sampler.set_epoch(epoch)
        counts.append(len(list(sampler)))

    average = sum(counts) / len(counts)
    assert average == pytest.approx(2.5, abs=0.1)


def test_sampler_varies_across_epochs():
    sampler = RepeatFactorSampler(image_repeats=[2.5], seed=0)
    lengths = set()
    for epoch in range(20):
        sampler.set_epoch(epoch)
        lengths.add(len(list(sampler)))
    assert lengths == {2, 3}  # both outcomes actually occur, not stuck on one


def test_sampler_len_matches_iteration_count():
    sampler = RepeatFactorSampler(image_repeats=[1.0, 2.5, 1.0], seed=0)
    sampler.set_epoch(3)
    assert len(sampler) == len(list(sampler))
