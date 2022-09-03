import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.autograd import Variable
from time import time
from split4models import *

# 载入MNIST数据集
train_set=torchvision.datasets.MNIST(root="./dataset",train=True,transform=transforms.ToTensor(),download=True)
# 加载网络
try:
    C=torch.load("./Models/MClient.pth").cuda()
except:
    C=CNet().cuda()
try:
    F=torch.load("./Models/F.pth").cuda()
except:
    F=tar().cuda()
try:
    invF=torch.load("./Models/invF.pth").cuda()
except:
    invF=invtar().cuda()
try:
    D=torch.load("./Models/D.pth").cuda()
except:
    D=Discriminator().cuda()
C.train()
F.train()
invF.train()
D.train()
# Dataloader和优化器
batch=64
Pubdataloader=DataLoader(train_set,batch_size=batch,shuffle=True)
Pridataloader=DataLoader(train_set,batch_size=batch,shuffle=True)

# 训练F和invF
optF=torch.optim.SGD(F.parameters(),lr=0.02)
optinvF=torch.optim.SGD(invF.parameters(),lr=0.02)
epoths=50
cri=nn.MSELoss()
for _ in range(epoths):
     for img,lab in Pubdataloader:
        img=img.cuda()
        optF.zero_grad()
        optinvF.zero_grad()
        # 正向传播
        output1=F(img)
        input2=Variable(output1.data,requires_grad=True)
        output2=invF(input2)
        # 计算损失函数和梯度
        loss=cri(output2,img)
        loss.backward()
        output1.backward(input2.grad)
        # 地图下降
        optF.step()
        optinvF.step()
        torch.save(F,"./Models/F.pth")
        torch.save(invF,"./Models/invF.pth")

# 训练分类器
epoths=50
optD=torch.optim.SGD(D.parameters(),lr=0.02)
cri=nn.CrossEntropyLoss()
for _ in range(epoths):
     for img,lab in Pubdataloader:
        img=img.cuda()
        optD.zero_grad()
        # 正向传播
        output1=C(img)
        output2=F(img)
        o1,o2=D(output1),D(output2)
        # 计算损失函数和梯度
        loss=cri(o1,torch.zeros(batch,dtype=torch.long).cuda())
        loss.backward()
        loss=cri(o2,torch.ones(batch,dtype=torch.long).cuda())
        loss.backward()
        # 梯度下降
        optD.step()
        torch.save(D,"./Models/D.pth")

# 特征空间劫持攻击
epoths=50
optC=torch.optim.SGD(C.parameters(),lr=0.02)
cri=nn.CrossEntropyLoss()
for _ in range(epoths):
     for img,lab in Pridataloader:
        img=img.cuda()
        optC.zero_grad()
        # 正向传播
        output1=C(img)
        input2=Variable(output1.data,requires_grad=True)
        o1=D(input2())
        # 计算损失函数和梯度
        loss=cri(o1,torch.zeros(batch,dtype=torch.long).cuda())
        loss.backward()
        output1.backward(input2.grad)
        # 梯度下降
        optC.step()
        torch.save(C,"./Models/MClient.pth")