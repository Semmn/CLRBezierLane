"""mmdet 3.x SingleStageDetector, reduced to its constructor and the train / test helpers FovealLaneNet uses, plus a
plain-PyTorch ResNet (BasicBlock, 34 layers by default) registered as "ResNet" for CPU tests."""
import torch.nn as nn
from mmdet.registry import MODELS


class SingleStageDetector(nn.Module):
    def __init__(self, backbone, neck=None, bbox_head=None, train_cfg=None, test_cfg=None, data_preprocessor=None,
                 init_cfg=None):
        super().__init__()
        self.backbone = MODELS.build(backbone)
        if neck is not None:
            self.neck = MODELS.build(neck)
        bbox_head = dict(bbox_head)
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = MODELS.build(bbox_head)
        self.train_cfg, self.test_cfg = train_cfg, test_cfg

    @property
    def with_neck(self):
        return hasattr(self, "neck") and self.neck is not None


class _Basic(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.down = (nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))
                     if stride != 1 or cin != cout else None)

    def forward(self, x):
        y = self.bn2(self.conv2(self.bn1(self.conv1(x)).relu()))
        return (y + (x if self.down is None else self.down(x))).relu()


@MODELS.register_module(name="ResNet")
class TinyResNet(nn.Module):
    """ResNet with BasicBlocks; depth 34 -> (3, 4, 6, 3) blocks; widths scaled by width (64 -> 64 * width)."""

    def __init__(self, depth=34, out_indices=(1, 2, 3), width=1.0, blocks=None, **unused):
        super().__init__()
        blocks = blocks or {18: (2, 2, 2, 2), 34: (3, 4, 6, 3)}[depth]
        ch = [int(c * width) for c in (64, 128, 256, 512)]
        self.stem = nn.Sequential(nn.Conv2d(3, ch[0], 7, 2, 3, bias=False), nn.BatchNorm2d(ch[0]), nn.ReLU(),
                                  nn.MaxPool2d(3, 2, 1))
        layers, cin = [], ch[0]
        for i, (n, c) in enumerate(zip(blocks, ch)):
            layers.append(nn.Sequential(*[_Basic(cin if j == 0 else c, c, (1 if i == 0 else 2) if j == 0 else 1)
                                          for j in range(n)]))
            cin = c
        self.layers = nn.ModuleList(layers)
        self.out_indices = tuple(out_indices)

    def forward(self, x):
        x = self.stem(x)
        outs = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)
