#服务端网络
from client import Res
import torch.nn.functional as F
import torch.nn as nn
import torch
class SNet(nn.Module):
    def __init__(self) -> None:
        super(SNet,self).__init__()
        self.res1=Res(128,128)
        self.res2=nn.Sequential(
            Res(128,256),
            Res(256,256),
            nn.Flatten()
        )
        self.lin=nn.Sequential(
            nn.Linear(50176,10),
            nn.Softmax(dim=1),
        )
    def forward(self,x):
        x=self.res1(x)
        x=self.res2(x)
        x=self.lin(x)
        return x
if __name__=='__main__':
    temp=SNet()
    input=torch.ones((10,128,14,14))
    output=temp(input)
    print(output)