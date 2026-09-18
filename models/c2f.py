import torch

class C2F_BLOCK(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x