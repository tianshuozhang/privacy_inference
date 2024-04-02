import torch
from models.Resnet18 import Resnet18
model=torch.load("./checkpoints/Resnet182024-03-27-13-42-56.pth").to('cpu')
example = torch.rand(1,3,32,32) # 注意，我这里导出的是CUDA版的模型，因为我的模型是在GPU中进行训练的
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
            if name is not None:
                setattr(self, f'layer{self.layer_num}_'+name, module)
            else :
                setattr(self, f'layer{self.layer_num}', module)
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

traced_script_module = torch.jit.trace(model1, example)
output = traced_script_module(torch.ones(1,3,32,32))
traced_script_module.save('./checkpoints/Resnet18.ptl')
print(traced_script_module)