import torch
from models.conv import ConvBlock
from models.c2f import C2fBlock

class Neck(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.upsample = torch.nn.Upsample(scale_factor=2, mode="nearest")

        # top-down (FPN)
        self.c2f_td_1 = C2fBlock(256 + 128, 128, 2, shortcut=False)
        self.c2f_td_2 = C2fBlock(128 + 64, 64, 2, shortcut=False)

        # bottom-up (PAN)
        self.down_1 = ConvBlock(64, 64, s=2)
        self.c2f_bu_1 = C2fBlock(64 + 128, 128, 2, shortcut=False)
        self.down_2 = ConvBlock(128, 128, s=2)
        self.c2f_bu_2 = C2fBlock(128 + 256, 256, 2, shortcut=False)

    def forward(self, p3, p4, p5):

        up1 = self.upsample(p5)
        cat1 = torch.cat([up1, p4], 1)
        n4 = self.c2f_td_1(cat1)

        up2 = self.upsample(n4)
        cat2 = torch.cat([up2, p3], 1)
        n3 = self.c2f_td_2(cat2)

        down1 = self.down_1(n3)
        cat3 = torch.cat([down1, n4], 1)
        n4b = self.c2f_bu_1(cat3)

        down2 = self.down_2(n4b)
        cat4 = torch.cat([down2, p5], 1)
        n5b = self.c2f_bu_2(cat4)

        return n3, n4b, n5b
