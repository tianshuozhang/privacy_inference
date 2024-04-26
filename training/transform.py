import torch
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
