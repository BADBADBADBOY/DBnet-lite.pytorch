#-*- coding:utf-8 _*-
"""
@author:fxw
@file: loss.py
@time: 2020/04/28
"""
import torch.nn as nn
from loss.dice_loss import DiceLoss,dice_loss
from loss.l1_loss import MaskL1Loss
from loss.balance_cross_entropy_loss import BalanceCrossEntropyLoss

class L1BalanceCELoss(nn.Module):
    '''
    Balanced CrossEntropy Loss on `binary`,
    MaskL1Loss on `thresh`,
    DiceLoss on `thresh_binary`.
    Note: The meaning of inputs can be figured out in `SegDetectorLossBuilder`.
    '''

    def __init__(self, eps=1e-6, l1_scale=10, bce_scale=1):
        super(L1BalanceCELoss, self).__init__()
        # self.dice_loss = DiceLoss(eps=eps)
        self.dice_loss = dice_loss
        self.l1_loss = MaskL1Loss()
        self.bce_loss = BalanceCrossEntropyLoss()
        self.l1_scale = l1_scale
        self.bce_scale = bce_scale

    def forward(self, pred, batch):
        bce_loss = self.bce_loss(pred['binary'], batch['gt'], batch['mask'])
        metrics = dict(bce_loss=bce_loss)
        if 'thresh' in pred:
            l1_loss, l1_metric = self.l1_loss(pred['thresh'], batch['thresh_map'], batch['thresh_mask'])
            dice_loss = self.dice_loss(pred['thresh_binary'], batch['gt'], batch['mask'])
            metrics['thresh_loss'] = dice_loss
            loss = dice_loss + self.l1_scale * l1_loss + bce_loss * self.bce_scale
            metrics.update(**l1_metric)
        else:
            loss = bce_loss
        return loss, metrics