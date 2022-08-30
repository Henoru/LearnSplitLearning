# Split Learning控制训练过程
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from client import CNet
from server import SNet
import torch.nn as nn
from torch.autograd import Variable
from time import time

# 载入MNIST数据集
train_set=torchvision.datasets.MNIST(root="./dataset",train=True,transform=transforms.ToTensor(),download=True)
test_set=torchvision.datasets.MNIST(root="./dataset",train=False,transform=transforms.ToTensor(),download=True)
# 数据集大小
train_set_size=len(train_set)
test_set_size=len(test_set)
print("Train set:",train_set_size)
print("Test set:",test_set_size)

# 加载数据
batch=64
train_dataloader=DataLoader(train_set,batch_size=batch,shuffle=True)

# 客户端模型
C=CNet().cuda()

# 服务端模型
S=SNet().cuda()

# 损失函数
criterion=nn.CrossEntropyLoss().cuda()

# 定义优化器
opt1=torch.optim.SGD(C.parameters(),lr=0.02)
opt2=torch.optim.SGD(S.parameters(),lr=0.02)

# 训练过程
start_time=time()
epoths=100

for _ in range(epoths):
    print("epoth:",_+1,end=" ")
    epoth_loss=0
    for img,lab in train_dataloader:
        img,lab=img.cuda(),lab.cuda()
        opt1.zero_grad()
        opt2.zero_grad()
        # 正向传播
        output1=C(img)
        input2=Variable(output1.data,requires_grad=True)
        output2=S(input2)
        # 计算损失函数和梯度
        loss=criterion(output2,lab)
        epoth_loss+=loss.item()
        loss.backward()
        output1.backward(input2.grad)
        #梯度下降
        opt1.step()
        opt2.step()
    print("Average loss:",epoth_loss/batch)

end_time=time()
print("time cost:",end_time-start_time)

# 测试集正确率
test_dataloader=DataLoader(test_set,batch_size=1000,shuffle=True)
correct=0
for img,lab in test_dataloader:
    img,lab=img.cuda(),lab.cuda()
    output=S(C(img))
    if torch.argmax(output)==lab:
        correct+=1
print("Model Accuracy =",correct/1000,"%")