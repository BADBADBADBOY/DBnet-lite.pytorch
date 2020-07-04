"""
#!-*- coding=utf-8 -*-
@author: BADBADBADBADBOY
@contact: 2441124901@qq.com
@software: PyCharm Community Edition
@file: FPN_Head.py
@time: 2020/7/4 15:16

"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBnRelu(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride,padding):
        super(ConvBnRelu,self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)  # Reduce channels
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class FPN_Head(nn.Module):
    def __init__(self,in_channels,inner_channels):
        super(FPN_Head,self).__init__()
        # Top layer
        self.toplayer = ConvBnRelu(in_channels[-1], inner_channels, kernel_size=1, stride=1, padding=0)  # Reduce channels
        # Smooth layers
        self.smooth1 = ConvBnRelu(inner_channels, inner_channels, kernel_size=3, stride=1, padding=1)
        self.smooth2 = ConvBnRelu(inner_channels, inner_channels, kernel_size=3, stride=1, padding=1)
        self.smooth3 = ConvBnRelu(inner_channels, inner_channels, kernel_size=3, stride=1, padding=1)
        # Lateral layers
        self.latlayer1 = ConvBnRelu(in_channels[-2], inner_channels, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = ConvBnRelu(in_channels[-3], inner_channels, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = ConvBnRelu(in_channels[-4], inner_channels, kernel_size=1, stride=1, padding=0)
        # Out map
        self.conv_out= nn.Conv2d(inner_channels*4, inner_channels, kernel_size=3, stride=1, padding=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.)
                m.bias.data.fill_(1e-4)

    def _upsample(self, x, y, scale=1):
        _, _, H, W = y.size()
        # return F.upsample(x, size=(H // scale, W // scale), mode='nearest')
        return F.interpolate(x, size=(H// scale, W// scale), mode='bilinear', align_corners=True)
    
    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        # return F.upsample(x, size=(H, W), mode='nearest') + y
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True) + y
    
    def forward(self, x):
        c2, c3, c4, c5 = x
        ##
        p5 = self.toplayer(c5)
        c4 = self.latlayer1(c4)
        p4 = self._upsample_add(p5, c4)
        p4 = self.smooth1(p4)
        c3 = self.latlayer2(c3)
        p3 = self._upsample_add(p4, c3)
        p3 = self.smooth2(p3)
        c2 = self.latlayer3(c2)
        p2 = self._upsample_add(p3, c2)
        p2 = self.smooth3(p2)
        ##
        p3 = self._upsample(p3, p2)
        p4 = self._upsample(p4, p2)
        p5 = self._upsample(p5, p2)

        out = torch.cat((p2, p3, p4, p5), 1)
        out = self.conv_out(out)
        return out
