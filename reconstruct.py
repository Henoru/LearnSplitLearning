import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


writer=SummaryWriter("./logs")

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
cnt=0
for img,lab in Pridataloader:
    cnt+=1
    img=img.cuda()
    output=invF(C(img))
    img=torch.squeeze(img,dim=0)
    output=torch.squeeze(output,dim=0)
    compare=torch.cat((img,output),dim=1)
    if cnt%1000==0:
        print(cnt)
    writer.add_image("results",compare,cnt)
writer.close()