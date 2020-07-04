"""
#!-*- coding=utf-8 -*-
@author: BADBADBADBADBOY
@contact: 2441124901@qq.com
@software: PyCharm Community Edition
@file: DBNet.py
@time: 2020/7/4 15:16

"""
import torch.nn as  nn
from models.head.seg_detector import SegDetector
from models.backbone.resnet import resnet18, resnet50,deformable_resnet50, deformable_resnet18

class DBNet(nn.Module):
    def __init__(self, config,is_train = True):
        super(DBNet, self).__init__()
        self.backbone = globals().get(config['train']['backbone'])(pretrained=config['train']['pretrained'])
        if(is_train is False):
            config['train']['adaptive'] = config['test']['adaptive']
        self.decode = SegDetector(headname = config['train']['HeadName'],
                                  in_channels = config['train']['in_channels'],
                                  inner_channels = config['train']['inner_channels'],
                                  k = config['train']['k'],
                                  bias=False, adaptive= config['train']['adaptive'], smooth=False, serial=False)
    def forward(self, x):
        x = self.backbone(x)
        out = self.decode.forward(x)
        return out


