import torch
from models.conv import ConvBlock
from models.bottleneck import Bottleneck

class C2fBlock(torch.nn.Module):

    def __init__(self, in_c, out_c, nb_bottlenecks, shortcut=True):
        super().__init__()
        c = out_c // 2
        self.in_conv = ConvBlock(in_c, 2*c, 1)
        self.nb_bottlenecks = nb_bottlenecks
        self.bottlenecks = torch.nn.ModuleList([Bottleneck(c, c, shortcut) for i in range(nb_bottlenecks)])
        self.out_conv = ConvBlock((2+nb_bottlenecks)*c, out_c, 1)

    def forward(self, x):

        x = self.in_conv(x)
        
        half_1, half_2 = torch.chunk(x, 2, 1)

        intermediaries = [half_1, half_2]
        last = half_1
        for b in self.bottlenecks:
            last = b(last)
            intermediaries.append(last)

        x = torch.cat(intermediaries, 1)

        x = self.out_conv(x)

        return x
