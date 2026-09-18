import torch
from bottleneck import Bottleneck

class Backbone(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):

        torch.nn.Conv2d(3, 32, (3,3), 2)

        torch.nn.Conv2d(32, 64, (3,3), 2)

        

        return x
