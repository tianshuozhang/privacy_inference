from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
from utils import forward_time
model=torch.load("./checkpoints/Lenet2024-01-14-21-35-48.pth")
test_dataset=datasets.MNIST("./data", train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(test_dataset,batch_size=32, shuffle=True)
forward_time(model=model,dataloader=test_loader,device='cuda')


