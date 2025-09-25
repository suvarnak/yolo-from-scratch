import torch.nn as nn
import torch 
from .backbone import Conv,C2f,yolo_params,Backbone
from .neck import Neck 
# DFL
class DFL(nn.Module):
    def __init__(self,ch=16):
        super().__init__()
        self.ch=ch
        self.conv=nn.Conv2d(in_channels=ch,out_channels=1,kernel_size=1,bias=False).requires_grad_(False)
        x=torch.arange(ch,dtype=torch.float).view(1,ch,1,1)
        self.conv.weight.data[:]=torch.nn.Parameter(x) # DFL only has ch parameters
    def forward(self,x):
        b,c,a=x.shape                           # c=4*ch
        x=x.view(b,4,self.ch,a).transpose(1,2)  # [bs,ch,4,a]
        x=x.softmax(1)                          # [b,ch,4,a]
        x=self.conv(x)                          # [b,1,4,a]
        return x.view(b,4,a)                    # [b,4,a]
class Head(nn.Module):
    def __init__(self,version,ch=16,num_classes=80):
        super().__init__()
        self.ch=ch                          # dfl channels
        self.coordinates=self.ch*4          # number of bounding box coordinates 
        self.nc=num_classes                 # 80 for COCO
        self.no=self.coordinates+self.nc    # number of outputs per anchor box
        self.stride=torch.zeros(3)          # strides computed during build
        d,w,r=yolo_params(version=version)
        self.box=nn.ModuleList([
            nn.Sequential(Conv(int(256*w),self.coordinates,kernel_size=3,stride=1,padding=1),
                          Conv(self.coordinates,self.coordinates,kernel_size=3,stride=1,padding=1),
                          nn.Conv2d(self.coordinates,self.coordinates,kernel_size=1,stride=1)),
            nn.Sequential(Conv(int(512*w),self.coordinates,kernel_size=3,stride=1,padding=1),
                          Conv(self.coordinates,self.coordinates,kernel_size=3,stride=1,padding=1),
                          nn.Conv2d(self.coordinates,self.coordinates,kernel_size=1,stride=1)),
            nn.Sequential(Conv(int(512*w*r),self.coordinates,kernel_size=3,stride=1,padding=1),
                          Conv(self.coordinates,self.coordinates,kernel_size=3,stride=1,padding=1),
                          nn.Conv2d(self.coordinates,self.coordinates,kernel_size=1,stride=1))
        ])
        self.cls=nn.ModuleList([
            nn.Sequential(Conv(int(256*w),self.nc,kernel_size=3,stride=1,padding=1),
                          Conv(self.nc,self.nc,kernel_size=3,stride=1,padding=1),
                          nn.Conv2d(self.nc,self.nc,kernel_size=1,stride=1)),
            nn.Sequential(Conv(int(512*w),self.nc,kernel_size=3,stride=1,padding=1),
                          Conv(self.nc,self.nc,kernel_size=3,stride=1,padding=1),
                          nn.Conv2d(self.nc,self.nc,kernel_size=1,stride=1)),
            nn.Sequential(Conv(int(512*w*r),self.nc,kernel_size=3,stride=1,padding=1),
                          Conv(self.nc,self.nc,kernel_size=3,stride=1,padding=1),
                          nn.Conv2d(self.nc,self.nc,kernel_size=1,stride=1))
        ])
        self.dfl=DFL()
    def forward(self,x):
        for i in range(len(self.box)):
            box=self.box[i](x[i])
            cls=self.cls[i](x[i])
            x[i]=torch.cat((box,cls),dim=1)
        if self.training:
            return x
        anchors, strides = (i.transpose(0, 1) for i in self.make_anchors(x, self.stride))
        x = torch.cat([i.view(x[0].shape[0], self.no, -1) for i in x], dim=2)
        box, cls = x.split(split_size=(4 * self.ch, self.nc), dim=1)
        a, b = self.dfl(box).chunk(2, 1)
        a = anchors.unsqueeze(0) - a
        b = anchors.unsqueeze(0) + b
        box = torch.cat(tensors=((a + b) / 2, b - a), dim=1)
        return torch.cat(tensors=(box * strides, cls.sigmoid()), dim=1)
    def make_anchors(self, x, strides, offset=0.5):
        assert x is not None
        anchor_tensor, stride_tensor = [], []
        dtype, device = x[0].dtype, x[0].device
        for i, stride in enumerate(strides):
            _, _, h, w = x[i].shape
            sx = torch.arange(end=w, device=device, dtype=dtype) + offset
            sy = torch.arange(end=h, device=device, dtype=dtype) + offset
            sy, sx = torch.meshgrid(sy, sx)
            anchor_tensor.append(torch.stack((sx, sy), -1).view(-1, 2))
            stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
        return torch.cat(anchor_tensor), torch.cat(stride_tensor)
if __name__=='__main__':
    dummy_input=torch.rand((1,64,128))
    dfl=DFL()
    print(f"{sum(p.numel() for p in dfl.parameters())} parameters")
    dummy_output=dfl(dummy_input)
    print(dummy_output.shape)
    print(dfl)
    detect=Head(version='n')
    print(f"{sum(p.numel() for p in detect.parameters())/1e6} million parameters")
    neck=Neck(version='n')
    print(f"{sum(p.numel() for p in neck.parameters())/1e6} million parameters")
    x=torch.rand((1,3,640,640))
    out1,out2,out3=Backbone(version='n')(x)
    out_1,out_2,out_3=neck(out1,out2,out3)
    print(out_1.shape)
    print(out_2.shape)
    print(out_3.shape)
    output=detect([out_1,out_2,out_3])
    print(output[0].shape)
    print(output[1].shape)
    print(output[2].shape)
    print(detect)
