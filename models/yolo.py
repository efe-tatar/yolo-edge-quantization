import torch

class YOLO(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x