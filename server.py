#服务端网络
from client import Res
import torch.nn.functional as F
import torch.nn as nn
class SNet(nn.Module):
    def __init__(self) -> None:
        super(SNet,self).__init__()
        self.res1=Res(128,128)
        self.res2=Res(128,256)
    def forward(self,x):
        x=self.res1(x)
        x=self.res2(x)
        return x
