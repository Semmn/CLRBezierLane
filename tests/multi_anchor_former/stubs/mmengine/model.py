import torch.nn as nn
class BaseModule(nn.Module):
    def __init__(self, init_cfg=None): super().__init__()
