import torch

class ConvBlock(torch.nn.Module):

    def __init__(self, in_c, out_c, k=3, s=1, act=torch.nn.SiLU):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_c, out_c, k, stride=s, padding=k//2, bias=False)
        self.bn = torch.nn.BatchNorm2d(out_c)
        self.act = act()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)

        return x
