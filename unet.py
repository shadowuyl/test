import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    '''out_h,out_w=in_h,in_w'''
    def __init__(self,in_ch,out_ch):
        super(DoubleConv,self).__init__()
        self.conv=nn.Sequential(
                nn.Conv2d(in_ch,out_ch,3,padding=1),
                nn.BatchNorm2d(out_ch)
                nn.ReLU(inplace=True)
                nn.Conv2d(out_ch,out_ch,3,padding=1),
                nn.BatchNorm2d(out_ch)
                nn.ReLU(inplace=True)
                )

    def forward(self,x):
        return self.conv(x)

class DownSample(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(DownSample,self).__init__()
        self.conv=nn.Sequential(
                nn.MaxPool2d(2),
                DoubleConv(in_ch,out_ch))

    def forward(self,x):
        return self.conv(x)

class UpSample(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(UpSample,self).__init__()
        self.up=nn.ConvTranspose2d(in_ch//2,in_ch//2,2,stride=2)
        self.conv=DoubleConv(in_ch,out_ch)
    
    def forward(self,x1,x2):
        x1=self.up(x1)
        dX=x1.size()[2]-x2.size()[2]
        dY=x1.size()[3]-x2.size()[3]
        x2=F.pad(x2,(dX//2,int(dX/2),dY//2,int(dY/2)))
        x=torch.cat([x2,x2],dim=1)
        x=self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(self,n_ch,n_cls):
        super(UNet,self).__init__()
        self.in=DoubleConv(n_ch,64)
        self.down1=DownSample(64,128)
        self.down2=DownSample(128,256)
        self.down3=DownSample(256,512)
        self.down4=DownSample(512,512)
        self.up1=UpSample(1024,256)
        self.up2=UpSample(512,128)
        self.up3=UpSample(256,64)
        self.up4=UpSample(128,64)
        self.out=nn.Conv2d(64,n_cls,1)

    def forward(self,x):
        x1=self.in(x)
        x2=self.down1(x1)
        x3=self.down2(x2)
        x4=self.down3(x3)
        x5=self.down4(x4)
        x=self.up1(x5,x4)
        x=self.up2(x,x3)
        x=self.up3(x,x2)
        x=self.up4(x,x1)
        x=self.out(x)
        return x

