"""
#!-*- coding=utf-8 -*-
@author: BADBADBADBADBOY
@contact: 2441124901@qq.com
@software: PyCharm Community Edition
@file: DB_Head.py
@time: 2020/7/4 15:16

"""
import torch
import torch.nn as nn

class ConvBnRelu(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride,padding,bias):
        super(ConvBnRelu,self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,bias=bias)  # Reduce channels
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class DB_Head(nn.Module):
    def __init__(self,in_channels,inner_channels,bias=False):
        super(DB_Head,self).__init__()

        self.up5 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up3 = nn.Upsample(scale_factor=2, mode='nearest')

        self.in5 = ConvBnRelu(in_channels[-1], inner_channels, 1,stride=1, padding=0, bias=bias)
        self.in4 = ConvBnRelu(in_channels[-2], inner_channels, 1,stride=1, padding=0, bias=bias)
        self.in3 = ConvBnRelu(in_channels[-3], inner_channels, 1, stride=1, padding=0,bias=bias)
        self.in2 = ConvBnRelu(in_channels[-4], inner_channels, 1,stride=1, padding=0, bias=bias)

        self.out5 = nn.Sequential(
            ConvBnRelu(inner_channels, inner_channels //4, 3,stride=1, padding=1, bias=bias),
            nn.Upsample(scale_factor=8, mode='nearest'))
        self.out4 = nn.Sequential(
            ConvBnRelu(inner_channels, inner_channels //4, 3, stride=1,padding=1, bias=bias),
            nn.Upsample(scale_factor=4, mode='nearest'))
        self.out3 = nn.Sequential(
            ConvBnRelu(inner_channels, inner_channels //4, 3, stride=1,padding=1, bias=bias),
            nn.Upsample(scale_factor=2, mode='nearest'))
        self.out2 = ConvBnRelu(inner_channels, inner_channels//4, 3, stride=1,padding=1, bias=bias)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.)
                m.bias.data.fill_(1e-4)

    def forward(self, x):
    
        c2, c3, c4, c5 = x
        in5 = self.in5(c5)
        in4 = self.in4(c4)
        in3 = self.in3(c3)
        in2 = self.in2(c2)

        out4 = self.up5(in5) + in4  # 1/16
        out3 = self.up4(out4) + in3  # 1/8
        out2 = self.up3(out3) + in2  # 1/4

        p5 = self.out5(in5)
        p4 = self.out4(out4)
        p3 = self.out3(out3)
        p2 = self.out2(out2)
        fuse = torch.cat((p5, p4, p3, p2), 1)
        return fuse