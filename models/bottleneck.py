import torch
from models.conv import ConvBlock

class Bottleneck(torch.nn.Module):

    def __init__(self, in_c, out_c, shortcut=True):
        super().__init__()

        self.apply_residual = (in_c == out_c) and shortcut

        self.conv1 = ConvBlock(in_c, out_c)
        self.conv2 = ConvBlock(out_c, out_c)

    def forward(self, x):

        resid = x

        x = self.conv1(x)
        x = self.conv2(x)

        if self.apply_residual:
            x = x + resid

        return x
