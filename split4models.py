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
            Res(128,128),
            Res(128,256),
            Res(256,256),
        )
    def forward(self,x):
        x=self.conv1(x)
        x=self.res1(x)
        x=self.res2(x)
        return x
class tar(nn.Module):
    def __init__(self) -> None:
        super(tar,self).__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(1,64,3,1,1),
        )
        self.conv2=nn.Sequential(
            nn.Conv2d(64,128,3,1,1),
            nn.Conv2d(128,256,3,1,1),
            nn.Conv2d(256,256,3,2,1),
        )
    def forward(self,x):
        x=self.conv1(x)
        x=self.conv2(x)
        return x
class invtar(nn.Module):
    def __init__(self) -> None:
        super(invtar,self).__init__()
        self.conv1=nn.Sequential(
            nn.ConvTranspose2d(256,256,3,1,1),
            nn.ConvTranspose2d(256,128,3,1,1),
            nn.ConvTranspose2d(128,1,3,2,1,output_padding=1),
        )
    def forward(self,x):
        x=self.conv1(x)
        return x
class Discriminator(nn.Module):
    def __init__(self) -> None:
        super(Discriminator,self).__init__()
        self.net=nn.Sequential(
            nn.Conv2d(256,128,3,1,1),
            Res(128,256),
            Res(256,256),
            Res(256,256),
            Res(256,256),
            Res(256,256),
            nn.Conv2d(256,1,3,2,2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64,2),
            nn.Softmax(dim=1),
        )
    def forward(self,x):
        x=self.net(x)
        return x
if __name__=='__main__':
    temp=Discriminator()
    temp2=tar()
    temp3=CNet()
    temp4=invtar()
    input=torch.ones((1,1,28,28))
    output1,output2=temp2(input),temp3(input)
    print(output1.shape)
    print(output2.shape)
    in2=temp4(output1)
    print(in2.shape)
    o1,o2=temp(output1),temp(output2)
    print(o1.shape)
    print(o2.shape)