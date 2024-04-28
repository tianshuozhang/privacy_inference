from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
from models.Lenet import Lenet
from models.Minionn import Minionn
from models.Sarda import Sarda
from models.SecureML import SecureML
from models.Resnet import ResNet18,ResNet34
from models.Resnet18 import Resnet18
from models.Resnet34 import Resnet34
# from models.vgg import vgg13
from models.vgg13 import vgg13
from utils import train,eval_acc,test_acc
from transform import FlattenModel

#training
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

transforms_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
train_dataset=datasets.CIFAR10("./data", train=True, transform=transform_train, download=True)
train_loader = DataLoader(train_dataset,batch_size=128, shuffle=True)
test_dataset=datasets.CIFAR10("./data", train=False, transform=transforms_test, download=True)
test_loader = DataLoader(test_dataset,batch_size=32, shuffle=True)

# model = vgg13()
model = torch.load("checkpoints/vgg132024-04-28-13-13-26.pth")

new_model=FlattenModel(model)
print(new_model)
 # 假设 src_model 是源模型， tgt_model 是目标模型
# 获取源模型的参数
src_state_dict = new_model.state_dict()
# 创建一个新的字典，其中的键是参数的名字，值是参数的值
new_state_dict = {}
for k, v in src_state_dict.items():
    new_state_dict[k] = v.detach().clone()

mymodel = vgg13()
# 加载新的参数到目标模型
mymodel.load_state_dict(new_state_dict)
print(mymodel)
# train(model=model,dataloader=train_loader,device="cuda",epochs=30,tag="vgg13")
test_acc(model=mymodel,dataloader=test_loader,device="cpu")

torch.save(mymodel,"checkpoints/vgg132024-04-28-13-13-26.pth")
