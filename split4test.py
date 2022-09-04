# Split Learning测试过程
from random import randint
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn

try:
    C=torch.load("./Models/MClient.pth").cuda()
except:
    print("No MClient Models")
try:
    invF=torch.load("./Models/invF.pth").cuda()
except:
    print("No invF Models")
train_set=torchvision.datasets.MNIST(root="./dataset",train=True,transform=transforms.ToTensor(),download=True)
Pridataloader=DataLoader(train_set,batch_size=1,shuffle=True)
C.eval()
invF.eval()
cri=nn.MSELoss()
loss=0
for img,lab in Pridataloader:
    img=img.cuda()
    guess=invF(C(img))
    loss+=cri(img,guess).item()
print(loss/len(train_set))
