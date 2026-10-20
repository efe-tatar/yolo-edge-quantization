import torch
from models.conv import ConvBlock

class SPPF(torch.nn.Module):

    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv_in = ConvBlock(in_c, in_c//2, 1)
        self.conv_out = ConvBlock(4*(in_c//2), out_c, 1)
        self.pool = torch.nn.MaxPool2d(5, 1, 2)

    def forward(self, x):

        x = self.conv_in(x)
        
        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)

        concat = torch.concat([x, y1, y2, y3], 1)

        x = self.conv_out(concat)

        return x
