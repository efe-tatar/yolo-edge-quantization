import torch
from models.c2f import C2fBlock
from models.conv import ConvBlock
from models.sppf import SPPF

class Head(torch.nn.Module):

    def __init__(self, c_in, num_classes):
        super().__init__()
        self.conv = ConvBlock(c_in, c_in, k=3)
        self.box = torch.nn.Conv2d(c_in, 4, kernel_size=1)
        self.cls = torch.nn.Conv2d(c_in, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        return self.box(x), self.cls(x)