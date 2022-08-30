# 客户端网络
import torch.nn.functional as F
import torch.nn as nn
import torch
class Res(nn.Module):
    def __init__(self,in_channels,out_channels) -> None:
        super(Res,self).__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(in_channels,out_channels,3,1,1),
            nn.BatchNorm2d(out_channels),
        )
        self.conv2=nn.Sequential(
            nn.Conv2d(out_channels,out_channels,3,1,1),
            nn.BatchNorm2d(out_channels),
        )
        self.conv3=nn.Sequential(
            nn.Conv2d(in_channels,out_channels,1,1,0),
            nn.BatchNorm2d(out_channels),
        )
    def forward(self,x):
        y=self.conv1(x)
        y=F.relu(y)
        y=self.conv2(y)
        x=self.conv3(x)
        y+=x
        return F.relu(y)
class CNet(nn.Module):
    def __init__(self) -> None:
        super(CNet,self).__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(1,64,3,1,1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
        )
        self.res1=Res(64,128)
        self.res2=nn.Sequential(
            Res(128,128),
            Res(128,128),
        )
    def forward(self,x):
        x=self.conv1(x)
        x=self.res1(x)
        x=self.res2(x)
        return x
if __name__=='__main__':
    temp=CNet()
    input=torch.ones((1,1,28,28))
    output=temp(input)
    print(output)