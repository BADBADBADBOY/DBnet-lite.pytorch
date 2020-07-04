"""
#!-*- coding=utf-8 -*-
@author: BADBADBADBADBOY
@contact: 2441124901@qq.com
@software: PyCharm Community Edition
@file: set_optimizer.py
@time: 2020/7/4 15:16

"""

def lr_poly(base_lr, epoch, max_epoch=1200, factor=0.9):
    return base_lr*((1-float(epoch)/max_epoch)**(factor))

def adjust_learning_rate_poly(base_lr, optimizer, epoch, max_epoch=1200, factor=0.9):
    lr = lr_poly(base_lr, epoch, max_epoch, factor)
    optimizer.param_groups[0]['lr'] = lr


def adjust_learning_rate(config, optimizer, epoch,gama = 0.1):
    if epoch in config['train']['schedule']:
        config['train']['base_lr'] =config['train']['base_lr'] * gama
        for param_group in optimizer.param_groups:
            param_group['lr'] = config['train']['base_lr']