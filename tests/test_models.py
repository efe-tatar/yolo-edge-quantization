import torch

from models.conv import ConvBlock
from models.bottleneck import Bottleneck
from models.c2f import C2fBlock
from models.sppf import SPPF
from models.backbone import Backbone
from models.neck import Neck
from models.head import Head
from models.yolo import YOLO


def n_params(module):
    return sum(p.numel() for p in module.parameters())


def test_convblock_shape_and_downsample():
    x = torch.randn(1, 3, 320, 320)
    m = ConvBlock(3, 16, k=3, s=2)
    out = m(x)
    assert out.shape == (1, 16, 160, 160)
    assert n_params(m) == 464


def test_convblock_fresh_activation_per_instance():
    a = ConvBlock(3, 8)
    b = ConvBlock(3, 8)
    assert a.act is not b.act


def test_bottleneck_shapes():
    x = torch.randn(1, 64, 40, 40)
    assert Bottleneck(64, 64)(x).shape == (1, 64, 40, 40)
    assert Bottleneck(64, 128)(x).shape == (1, 128, 40, 40)
    assert n_params(Bottleneck(64, 64)) == 73984


def test_bottleneck_shortcut_flag():
    assert Bottleneck(64, 64, shortcut=True).apply_residual is True
    assert Bottleneck(64, 64, shortcut=False).apply_residual is False
    assert Bottleneck(64, 128, shortcut=True).apply_residual is False


def test_c2f_shapes():
    x = torch.randn(1, 64, 40, 40)
    m = C2fBlock(64, 64, 2)
    assert m(x).shape == (1, 64, 40, 40)
    assert n_params(m) == 49664
    assert C2fBlock(64, 128, 2)(x).shape == (1, 128, 40, 40)
    assert C2fBlock(64, 64, 1, shortcut=False)(x).shape == (1, 64, 40, 40)


def test_sppf_shape():
    x = torch.randn(1, 256, 10, 10)
    m = SPPF(256, 256)
    assert m(x).shape == (1, 256, 10, 10)


def test_backbone_shapes():
    x = torch.randn(1, 3, 320, 320)
    p3, p4, p5 = Backbone()(x)
    assert p3.shape == (1, 64, 40, 40)
    assert p4.shape == (1, 128, 20, 20)
    assert p5.shape == (1, 256, 10, 10)


def test_head_shapes():
    x = torch.randn(1, 128, 20, 20)
    box, cls = Head(128, num_classes=54)(x)
    assert box.shape == (1, 4, 20, 20)
    assert cls.shape == (1, 54, 20, 20)


def test_neck_shapes():
    p3 = torch.randn(1, 64, 40, 40)
    p4 = torch.randn(1, 128, 20, 20)
    p5 = torch.randn(1, 256, 10, 10)
    n3, n4, n5 = Neck()(p3, p4, p5)
    assert n3.shape == (1, 64, 40, 40)
    assert n4.shape == (1, 128, 20, 20)
    assert n5.shape == (1, 256, 10, 10)


def test_yolo_end_to_end_shapes():
    x = torch.randn(1, 3, 320, 320)
    num_classes = 54
    (box3, cls3), (box4, cls4), (box5, cls5) = YOLO(num_classes)(x)

    assert box3.shape == (1, 4, 40, 40)
    assert cls3.shape == (1, num_classes, 40, 40)

    assert box4.shape == (1, 4, 20, 20)
    assert cls4.shape == (1, num_classes, 20, 20)

    assert box5.shape == (1, 4, 10, 10)
    assert cls5.shape == (1, num_classes, 10, 10)
