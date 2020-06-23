# -*- coding:utf-8 _*-
"""
@author:fxw
@file: DBNet.py
@time: 2020/04/28
"""
import torch.nn as  nn
from models.resnet import deformable_resnet18, resnet18, resnet50, deformable_resnet50
from models.seg_detector import SegDetector


class DBNet(nn.Module):
    def __init__(self, model_name, adaptive):
        super(DBNet, self).__init__()
        self.backbone = globals().get(model_name)()
        if ('18' in model_name):
            in_channels = [64, 128, 256, 512]
        else:
            in_channels = [256, 512, 1024, 2048]

        self.decode = SegDetector(in_channels=in_channels,
                                  inner_channels=256, k=50,
                                  bias=False, adaptive=adaptive, smooth=False, serial=False)

    def forward(self, x):
        x = self.backbone(x)
        out = self.decode.forward(x)
        return out


