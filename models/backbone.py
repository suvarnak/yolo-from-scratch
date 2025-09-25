import torch.nn as nn
import torch

__all__ = [
    "Conv", "Bottleneck", "C2f", "SPPF", "yolo_params", "Backbone"
]

class Conv(nn.Module):
    def __init__(self,in_channels, out_channels,kernel_size=3,stride=1,padding=1,groups=1,activation=True):
        super().__init__()
        self.conv=nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding,bias=False,groups=groups)
        self.bn=nn.BatchNorm2d(out_channels,eps=0.001,momentum=0.03)
        self.act=nn.SiLU(inplace=True) if activation else nn.Identity()

    def forward(self,x):
        return self.act(self.bn(self.conv(x)))

class Bottleneck(nn.Module):
    def __init__(self,in_channels,out_channels,shortcut=True):
        super().__init__()
        self.conv1=Conv(in_channels,out_channels,kernel_size=3,stride=1,padding=1)
        self.conv2=Conv(out_channels,out_channels,kernel_size=3,stride=1,padding=1)
        self.shortcut=shortcut

    def forward(self,x):
        x_in=x # for residual connection
        x=self.conv1(x)
        x=self.conv2(x)
        if self.shortcut:
            x=x+x_in
        return x

class C2f(nn.Module):
    def __init__(self,in_channels,out_channels, num_bottlenecks,shortcut=True):
        super().__init__()
        self.mid_channels=out_channels//2
        self.num_bottlenecks=num_bottlenecks
        self.conv1=Conv(in_channels,out_channels,kernel_size=1,stride=1,padding=0)
        self.m=nn.ModuleList([Bottleneck(self.mid_channels,self.mid_channels) for _ in range(num_bottlenecks)])
        self.conv2=Conv((num_bottlenecks+2)*out_channels//2,out_channels,kernel_size=1,stride=1,padding=0)
    def forward(self,x):
        x=self.conv1(x)
        x1,x2=x[:,:x.shape[1]//2,:,:], x[:,x.shape[1]//2:,:,:]
        outputs=[x1,x2]
        for i in range(self.num_bottlenecks):
            x1=self.m[i](x1)
            outputs.insert(0,x1)
        outputs=torch.cat(outputs,dim=1)
        out=self.conv2(outputs)
        return out

class SPPF(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=5):
        super().__init__()
        hidden_channels=in_channels//2
        self.conv1=Conv(in_channels,hidden_channels,kernel_size=1,stride=1,padding=0)
        self.conv2=Conv(4*hidden_channels,out_channels,kernel_size=1,stride=1,padding=0)
        self.m=nn.MaxPool2d(kernel_size=kernel_size,stride=1,padding=kernel_size//2,dilation=1,ceil_mode=False)
    def forward(self,x):
        x=self.conv1(x)
        y1=self.m(x)
        y2=self.m(y1)
        y3=self.m(y2)
        y=torch.cat([x,y1,y2,y3],dim=1)
        y=self.conv2(y)
        return y

def yolo_params(version):
    if version=='n':
        return 1/3,1/4,2.0
    elif version=='s':
        return 1/3,1/2,2.0
    elif version=='m':
        return 2/3,3/4,1.5
    elif version=='l':
        return 1.0,1.0,1.0
    elif version=='x':
        return 1.0,1.25,1.0
    else:
        raise ValueError(f"Unknown YOLO version: {version}")

class Backbone(nn.Module):
    def __init__(self,version,in_channels=3,shortcut=True):
        super().__init__()
        d,w,r=yolo_params(version)
        self.conv_0=Conv(in_channels,int(64*w),kernel_size=3,stride=2,padding=1)
        self.conv_1=Conv(int(64*w),int(128*w),kernel_size=3,stride=2,padding=1)
        self.conv_3=Conv(int(128*w),int(256*w),kernel_size=3,stride=2,padding=1)
        self.conv_5=Conv(int(256*w),int(512*w),kernel_size=3,stride=2,padding=1)
        self.conv_7=Conv(int(512*w),int(512*w*r),kernel_size=3,stride=2,padding=1)
        self.c2f_2=C2f(int(128*w),int(128*w),num_bottlenecks=int(3*d),shortcut=True)
        self.c2f_4=C2f(int(256*w),int(256*w),num_bottlenecks=int(6*d),shortcut=True)
        self.c2f_6=C2f(int(512*w),int(512*w),num_bottlenecks=int(6*d),shortcut=True)
        self.c2f_8=C2f(int(512*w*r),int(512*w*r),num_bottlenecks=int(3*d),shortcut=True)
        self.sppf=SPPF(int(512*w*r),int(512*w*r))
    def forward(self,x):
        x=self.conv_0(x)
        x=self.conv_1(x)
        x=self.c2f_2(x)
        x=self.conv_3(x)
        out1=self.c2f_4(x)
        x=self.conv_5(out1)
        out2=self.c2f_6(x)
        x=self.conv_7(out2)
        x=self.c2f_8(x)
        out3=self.sppf(x)
        return out1,out2,out3
