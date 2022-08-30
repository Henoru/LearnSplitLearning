import torch
import torchvision
print(torch.__version__)
train_set=torchvision.datasets.MNIST(root="./dataset",train=True,download=True)
print(train_set[0])