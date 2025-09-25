import torch.nn as nn
import torch
from .backbone import Conv, C2f, yolo_params, Backbone
from .head import Head
from .neck import Neck

class MyYolo(nn.Module):
    def __init__(self,version):
        super().__init__()
        self.backbone=Backbone(version=version)
        self.neck=Neck(version=version)
        self.head=Head(version=version)

    def forward(self,x):
        x=self.backbone(x)              # return out1,out2,out3
        x=self.neck(x[0],x[1],x[2])     # return out_1, out_2,out_3
        return self.head(list(x))
if __name__=='__main__':
    model=MyYolo(version='n')
    print(f"{sum(p.numel() for p in model.parameters())/1e6} million parameters")
    print(model)
