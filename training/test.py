from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
from models.Lenet import Lenet
from models.Minionn import Minionn
from models.Sarda import Sarda
from models.SecureML import SecureML
from models.Resnet import ResNet18
from models.Resnet18 import Resnet18
from utils import train,eval_acc,test_acc
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

model=torch.load("./checkpoints/Resnet182024-03-27-13-42-56.pth").to('cpu')

model1=Resnet18()
model = model.eval()
class FlattenModel(torch.nn.Module):
    def __init__(self, original_model):
        super(FlattenModel, self).__init__()
        self.module_list = []  # 把这里改成python的list。
        self.layer_num = 1
        self.flatten_module(original_model)

    def flatten_module(self, module,name=None):
        if len(list(module.children())) == 0:

            # 复制 old_layer 的权重到 module
            if name is not None:
                setattr(self, f'layer{self.layer_num}_'+name, module)
                self.__getattr__(f'layer{self.layer_num}_'+name).load_state_dict(module.state_dict())
            else :
                setattr(self, f'layer{self.layer_num}', module)
                self.__getattr__(f'layer{self.layer_num}').load_state_dict(module.state_dict())

            
            self.module_list.append(module)
            self.layer_num += 1
        else:
            for key,child in module.named_children():
                if "shortcut" in key:
                    self.flatten_module(child,"shortcut")
                else:
                    self.flatten_module(child,name)

new_model=FlattenModel(model)
 # 假设 src_model 是源模型， tgt_model 是目标模型

# 获取源模型的参数
src_state_dict = new_model.state_dict()

# 创建一个新的字典，其中的键是参数的名字，值是参数的值
new_state_dict = {}

for k, v in src_state_dict.items():
    # .detach() 方法可以使得参数脱离它们所在的计算图，使得参数的修改不会影响到原计算图
    # .clone() 方法可以复制参数
    new_state_dict[k] = v.detach().clone()

# 加载新的参数到目标模型
model1.load_state_dict(new_state_dict)
# train(model=model,dataloader=train_loader,device="cuda",epochs=30,tag="Resnet18")
# test_acc(model=model1,dataloader=test_loader,device="cpu")
torch.save(model1,"./checkpoints/Resnet182024-03-27-13-42-56.pth")