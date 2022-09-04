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
# Setup Phase
optF=torch.optim.SGD(F.parameters(),lr=0.02)
optinvF=torch.optim.SGD(invF.parameters(),lr=0.02)
optD=torch.optim.SGD(D.parameters(),lr=0.02)
epochs=100
cri1=nn.MSELoss()
cri2=nn.CrossEntropyLoss()
for _ in range(epochs):
    tot_loss=0
    cnt=0
    print("epoch:",_+1,end=" ")
    for img,lab in Pubdataloader:
        img=img.cuda()
        optF.zero_grad()
        optinvF.zero_grad()
        optD.zero_grad()
        # 正向传播
        output1=F(img)
        output2=C(img)
        input1=Variable(output1.data,requires_grad=True)
        input2=Variable(output2.data,requires_grad=True)
        infer=invF(input1)
        o1,o2=D(input1),D(input2)
        # 计算损失函数和梯度
        loss=cri1(infer,img)+cri2(o1,torch.ones(o1.shape[0],dtype=torch.long).cuda())+cri2(o2,torch.zeros(o2.shape[0],dtype=torch.long).cuda())
        tot_loss+=loss.item()
        loss.backward()
        output1.backward(input1.grad)
        output2.backward(input2.grad)
        # 地图下降
        optF.step()
        optinvF.step()
        optD.step()
    print(tot_loss/cnt)
    torch.save(F,"./Models/F.pth")
    torch.save(invF,"./Models/invF.pth")
    torch.save(D,"./Models/D.pth")
    
# Client’s training procedure
epochs=100
optC=torch.optim.SGD(C.parameters(),lr=0.02)
cri=nn.CrossEntropyLoss()
for _ in range(epochs):
    tot_loss=0
    cnt=0
    print("epoch:",_+1,end=" ")
    for img,lab in Pridataloader:
        img=img.cuda()
        optC.zero_grad()
        # 正向传播
        output1=C(img)
        input2=Variable(output1.data,requires_grad=True)
        o1=D(input2)
        # 计算损失函数和梯度
        loss=cri(o1,torch.ones(o1.shape[0],dtype=torch.long).cuda())
        tot_loss+=loss.item()
        cnt+=1
        loss.backward()
        output1.backward(input2.grad)
        # 梯度下降
        optC.step()
    print(tot_loss/cnt)
    torch.save(C,"./Models/MClient.pth")