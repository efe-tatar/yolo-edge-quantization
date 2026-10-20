import torch
from models.c2f import C2fBlock
from models.conv import ConvBlock
from models.sppf import SPPF

class Backbone(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.stem_1 = ConvBlock(3, 16, s=2)
        self.stem_2 = ConvBlock(16, 32, s=2)
        self.stage_1 = C2fBlock(32, 32, 1)
        self.down_2 = ConvBlock(32, 64, s=2)
        self.stage_2 = C2fBlock(64,64, 2)
        self.down_3 = ConvBlock(64, 128, s=2)
        self.stage_3 = C2fBlock(128, 128, 2)
        self.down_4 = ConvBlock(128, 256, s=2)
        self.stage_4 = C2fBlock(256, 256, 1)
        self.sppf = SPPF(256, 256)

    def forward(self, x):

        x = self.stem_1(x)
        x = self.stem_2(x)
        x = self.stage_1(x)
        x = self.down_2(x)
        p3 = self.stage_2(x)
        x = self.down_3(p3)
        p4 = self.stage_3(x)
        x = self.down_4(p4)
        x = self.stage_4(x)
        p5 = self.sppf(x)

        return p3, p4, p5
