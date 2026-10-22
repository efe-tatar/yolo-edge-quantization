import math

import torch

from models.yolo import YOLO
from train.schedule import param_groups, warmup_cosine


def test_warmup_rises_linearly_to_base_lr():
    lr_at = warmup_cosine(base_lr=1e-3, epochs=50, warmup_epochs=4)

    assert lr_at(0) == 0.0
    assert math.isclose(lr_at(2), 0.5e-3)
    assert math.isclose(lr_at(4), 1e-3)


def test_cosine_decays_to_final_lr_and_stays_there():
    lr_at = warmup_cosine(base_lr=1e-3, epochs=50, warmup_epochs=4, final_lr_ratio=0.01)

    values = [lr_at(t) for t in range(4, 51)]
    assert all(a >= b for a, b in zip(values, values[1:]))  # monotonically non-increasing
    assert math.isclose(lr_at(50), 1e-5)
    assert math.isclose(lr_at(60), 1e-5)  # past the end: clamped, not wrapping around
    assert math.isclose(lr_at(27), (1e-3 + 1e-5) / 2)  # halfway through the decay


def test_no_warmup():
    lr_at = warmup_cosine(base_lr=1e-3, epochs=10, warmup_epochs=0)
    assert math.isclose(lr_at(0), 1e-3)


def test_param_groups_exclude_biases_and_batchnorm_from_decay():
    model = YOLO(3)
    decay, no_decay = param_groups(model, weight_decay=0.05)

    assert decay["weight_decay"] == 0.05 and no_decay["weight_decay"] == 0.0
    assert all(p.ndim > 1 for p in decay["params"])
    assert all(p.ndim == 1 for p in no_decay["params"])
    assert len(decay["params"]) + len(no_decay["params"]) == len(list(model.parameters()))

    torch.optim.AdamW([decay, no_decay], lr=1e-3)  # accepted as-is
