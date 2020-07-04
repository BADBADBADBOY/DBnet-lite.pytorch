"""
#!-*- coding=utf-8 -*-
@author: BADBADBADBADBOY
@contact: 2441124901@qq.com
@software: PyCharm Community Edition
@file: fuse_model.py
@time: 2020/7/4 15:16

"""
import torch
import torch.nn as nn
import time
import sys
import numpy as np
import torchvision
import torch.nn.functional as F


class DummyModule(nn.Module):
    def __init__(self):
        super(DummyModule, self).__init__()

    def forward(self, x):
        return x

def fuse(conv, bn):
    # *******************conv参数********************
    w = conv.weight

    # ********************BN参数*********************
    mean = bn.running_mean
    var_sqrt = torch.sqrt(bn.running_var + bn.eps)
    gamma = bn.weight
    beta = bn.bias

    if conv.bias is not None:
        b = conv.bias
    else:
        b = mean.new_zeros(mean.shape)

    w = w * (gamma / var_sqrt).reshape([conv.out_channels, 1, 1, 1])
    b = (b - mean)/var_sqrt * gamma + beta

    fused_conv = nn.Conv2d(conv.in_channels,
                         conv.out_channels,
                         conv.kernel_size,
                         conv.stride,
                         conv.padding,
                         bias=True)
    fused_conv.weight = nn.Parameter(w)
    fused_conv.bias = nn.Parameter(b)
    return fused_conv

def fuse_module(m):
    children = list(m.named_children())
    c = None
    cn = None
    for name, child in children:
        if isinstance(child, nn.BatchNorm2d) and c is not None:
            bc = fuse(c, child)
            m._modules[cn] = bc
            m._modules[name] = DummyModule()
            c = None
        elif isinstance(child, nn.Conv2d):
            c = child
            cn = name
        else:
            fuse_module(child)
            