from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
from models.Lenet import Lenet
from models.Minionn import Minionn
from models.Sarda import Sarda
from models.SecureML import SecureML
from utils import train,eval_acc,test_acc
#training
train_dataset=datasets.MNIST("./data", train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(train_dataset,batch_size=128, shuffle=True)
test_dataset=datasets.MNIST("./data", train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(test_dataset,batch_size=32, shuffle=True)
model=Lenet()
train(model=model,dataloader=train_loader,device="cuda" if torch.cuda.is_available() else "cpu",tag='Lenet')
test_acc(model=model,dataloader=test_loader)
model1=Minionn()
train(model=model1,dataloader=train_loader,device="cuda" if torch.cuda.is_available() else "cpu",tag='Minionn')
test_acc(model=model1,dataloader=test_loader)
model2=Sarda()
train(model=model2,dataloader=train_loader,device="cuda" if torch.cuda.is_available() else "cpu",tag='Sarda')
test_acc(model=model2,dataloader=test_loader)
