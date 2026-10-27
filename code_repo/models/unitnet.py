"""Models expressed as a sequence of units (the granularity for stages, freezing, shadows).

Interface used by the trainers, profiler and emulator:
    net.units                       nn.ModuleList
    net.prepare(batch) -> (x, ctx)  first unit input and side info (labels, masks)
    net.unit_forward(i, x, ctx)     output of unit i; the LAST unit returns the scalar loss
    net.loss(batch, on_unit=None)   full forward; on_unit(i) is called before unit i runs
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class UnitNet(nn.Module):
    def loss(self, batch, on_unit=None):
        x, ctx = self.prepare(batch)
        for i in range(len(self.units)):
            if on_unit is not None:
                on_unit(i)
            x = self.unit_forward(i, x, ctx)
        return x

    def unit_params(self):
        return [list(u.parameters()) for u in self.units]


# ============================================================================ ResNets (CIFAR)
class BasicBlock(nn.Module):
    """He et al. CIFAR basic block; projection (1x1 conv + BN) shortcut when shape changes."""

    def __init__(self, cin, cout, stride):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm2d(cout)
        self.short = None
        if stride != 1 or cin != cout:
            self.short = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        return F.relu(y + (x if self.short is None else self.short(x)))


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, cin, planes, stride):
        super().__init__()
        cout = planes * 4
        self.c1 = nn.Conv2d(cin, planes, 1, bias=False)
        self.b1 = nn.BatchNorm2d(planes)
        self.c2 = nn.Conv2d(planes, planes, 3, stride, 1, bias=False)
        self.b2 = nn.BatchNorm2d(planes)
        self.c3 = nn.Conv2d(planes, cout, 1, bias=False)
        self.b3 = nn.BatchNorm2d(cout)
        self.short = None
        if stride != 1 or cin != cout:
            self.short = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)))
        y = F.relu(self.b2(self.c2(y)))
        y = self.b3(self.c3(y))
        return F.relu(y + (x if self.short is None else self.short(x)))


class Head(nn.Module):
    def __init__(self, cin, classes):
        super().__init__()
        self.fc = nn.Linear(cin, classes)

    def forward(self, x):
        return self.fc(torch.flatten(F.adaptive_avg_pool2d(x, 1), 1))


class ResNetUnits(UnitNet):
    def __init__(self, kind, classes, zero_init_residual=False):
        super().__init__()
        units = []
        if kind in ("resnet56", "resnet110"):
            n = {"resnet56": 9, "resnet110": 18}[kind]
            units.append(nn.Sequential(nn.Conv2d(3, 16, 3, 1, 1, bias=False), nn.BatchNorm2d(16), nn.ReLU()))
            cin = 16
            for cout, stride in ((16, 1), (32, 2), (64, 2)):
                for b in range(n):
                    units.append(BasicBlock(cin, cout, stride if b == 0 else 1))
                    cin = cout
        elif kind == "resnet50":
            # CIFAR variant: 3x3 stem, no max-pool (as in common CIFAR ResNet-50 code)
            units.append(nn.Sequential(nn.Conv2d(3, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU()))
            cin = 64
            for planes, nblk, stride in ((64, 3, 1), (128, 4, 2), (256, 6, 2), (512, 3, 2)):
                for b in range(nblk):
                    units.append(Bottleneck(cin, planes, stride if b == 0 else 1))
                    cin = planes * 4
        else:
            raise ValueError(kind)
        units.append(Head(cin, classes))
        self.units = nn.ModuleList(units)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
        if zero_init_residual:      # each residual block starts as the identity (Goyal et al. 2017)
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.zeros_(m.b3.weight)
                elif isinstance(m, BasicBlock):
                    nn.init.zeros_(m.b2.weight)

    def prepare(self, batch):
        return batch[0], batch[1]

    def unit_forward(self, i, x, ctx):
        x = self.units[i](x)
        if i == len(self.units) - 1:
            return F.cross_entropy(x, ctx)
        return x

    def logits(self, x):
        for u in self.units:
            x = u(x)
        return x
