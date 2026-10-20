import torch
from models.backbone import Backbone
from models.neck import Neck
from models.head import Head

class YOLO(torch.nn.Module):

    def __init__(self, num_classes):
        super().__init__()
        self.backbone = Backbone()
        self.neck = Neck()
        self.head_p3 = Head(64, num_classes)
        self.head_p4 = Head(128, num_classes)
        self.head_p5 = Head(256, num_classes)

    def forward(self, x):
        p3, p4, p5 = self.backbone(x)
        n3, n4, n5 = self.neck(p3, p4, p5)
        return self.head_p3(n3), self.head_p4(n4), self.head_p5(n5)
