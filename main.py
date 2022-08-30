# Split Learning控制训练过程
import torch
import torchvision
import torchvision.transforms as transforms
# 载入MNIST训练数据
train_set=torchvision.datasets.MNIST(root="./dataset",train=True,transform=transforms.ToTensor(),download=True)
