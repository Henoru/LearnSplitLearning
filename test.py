# Split Learning测试过程
from random import randint
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# 载入MNIST数据集
test_set=torchvision.datasets.MNIST(root="./dataset",train=False,transform=transforms.ToTensor(),download=True)
# 数据集大小
test_set_size=len(test_set)
print("Test set:",test_set_size)

# 客户端模型
try:
    C=torch.load("./Models/Client.pth").cuda()
except:
    print("NO Client Models")
    exit()

# 服务端模型
try:
    S=torch.load("./Models/Server.pth").cuda()
except:
    print("NO Server Models")
    exit()
# 测试集正确率
C.eval() 
S.eval()
correct=0
with torch.no_grad():
    for _ in range(1000):
        data=test_set[randint(0,test_set_size-1)]
        img,lab=data[0].cuda(),data[1]
        output=S(C(img.unsqueeze(0)))
        if torch.argmax(output).item()==lab:
            correct+=1
print("Model Accuracy =",correct/10,"%")