import torch
from models.Resnet18 import Resnet18
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from utils import test_acc

model=torch.load("./checkpoints/Resnet342024-04-22-20-02-58.pth").to('cpu')

example = torch.rand(1,3,32,32) # 注意，我这里导出的是CUDA版的模型，因为我的模型是在GPU中进行训练的
traced_script_module = torch.jit.trace(model, example)
output = traced_script_module(torch.ones(1,3,32,32))
traced_script_module.save('./checkpoints/Resnet34.ptl')
transforms_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

test_dataset=datasets.CIFAR10("./data", train=False, transform=transforms_test, download=True)
test_loader = DataLoader(test_dataset,batch_size=32, shuffle=True)

test_acc(model=traced_script_module,dataloader=test_loader,device="cpu")
