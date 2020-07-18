"""
#!-*- coding=utf-8 -*-
@author: BADBADBADBADBOY
@contact: 2441124901@qq.com
@software: PyCharm Community Edition
@file: tt.py
@time: 2020/6/20 10:51

"""

import models
import torch
import torch.nn as nn
import numpy as np


def get_new_model(model, new_model, prued_mask, bn_index):
    merge1_index = [3, 12, 18]
    merge2_index = [25, 28, 34]
    merge3_index = [41, 44, 50]
    merge4_index = [57, 60, 66]

    index_0 = []
    for item in merge1_index:
        index_0.append(bn_index.index(item))
    mask1 = prued_mask[index_0[0]] | prued_mask[index_0[1]] | prued_mask[index_0[2]]

    index_1 = []
    for item in merge2_index:
        index_1.append(bn_index.index(item))
    mask2 = prued_mask[index_1[0]] | prued_mask[index_1[1]] | prued_mask[index_1[2]]

    index_2 = []
    for item in merge3_index:
        index_2.append(bn_index.index(item))
    mask3 = prued_mask[index_2[0]] | prued_mask[index_2[1]] | prued_mask[index_2[2]]

    index_3 = []
    for item in merge4_index:
        index_3.append(bn_index.index(item))
    mask4 = prued_mask[index_3[0]] | prued_mask[index_3[1]] | prued_mask[index_3[2]]

    for index in index_0:
        prued_mask[index] = mask1

    for index in index_1:
        prued_mask[index] = mask2

    for index in index_2:
        prued_mask[index] = mask3

    for index in index_3:
        prued_mask[index] = mask4

    ##############################################################
    index_bn = 0
    index_conv = 0

    bn_mask = []
    conv_in_mask = []
    conv_out_mask = []
    tag = 0
    for m in new_model.modules():
        if (tag > 69):
            break
        if (isinstance(m, nn.BatchNorm2d)):
            m.num_features = prued_mask[index_bn].sum()
            bn_mask.append(prued_mask[index_bn])
            index_bn += 1
        elif (isinstance(m, nn.Conv2d)):
            if (index_conv == 0):
                m.in_channels = 3
                conv_in_mask.append(torch.ones(3))
            else:
                m.in_channels = prued_mask[index_conv - 1].sum()
                conv_in_mask.append(prued_mask[index_conv - 1])
            m.out_channels = int(prued_mask[index_conv].sum())
            conv_out_mask.append(prued_mask[index_conv])
            index_conv += 1
        tag += 1

    conv_change_index = [27, 43, 59]  #
    change_conv_bn_index = [18, 34, 50]  #
    tag = 0
    for m in new_model.modules():
        if (tag > 69):
            break
        if (isinstance(m, nn.Conv2d)):
            if (tag in conv_change_index):
                index = conv_change_index.index(tag)
                index = change_conv_bn_index[index]
                index = bn_index.index(index)
                mask = prued_mask[index]
                conv_in_mask[index + 3] = mask
                m.in_channels = mask.sum()
        tag += 1

    #############################################################
    bn_i = 0
    conv_i = 0
    scale_i = 0
    scale_mask = [mask4, mask3, mask2, mask1]
    #     scale = [70,86,90,94]  # FPN
    scale = [73, 77, 81, 85]  # DB
    for [m0, m1] in zip(model.modules(), new_model.modules()):
        if (scale_i > 69):
            if isinstance(m0, nn.Conv2d):
                if (scale_i in scale):
                    index = scale.index(scale_i)
                    m1.in_channels = scale_mask[index].sum()
                    idx0 = np.squeeze(np.argwhere(np.asarray(scale_mask[index].cpu().numpy())))
                    idx1 = np.squeeze(np.argwhere(np.asarray(torch.ones(256).cpu().numpy())))
                    if idx0.size == 1:
                        idx0 = np.resize(idx0, (1,))
                    if idx1.size == 1:
                        idx1 = np.resize(idx1, (1,))
                    w = m0.weight.data[:, idx0, :, :].clone()
                    m1.weight.data = w[idx1, :, :, :].clone()
                    if m1.bias is not None:
                        m1.bias.data = m0.bias.data[idx1].clone()

                else:
                    m1.weight.data = m0.weight.data.clone()
                    if m1.bias is not None:
                        m1.bias.data = m0.bias.data.clone()

            elif isinstance(m0, nn.BatchNorm2d):
                m1.weight.data = m0.weight.data.clone()
                if m1.bias is not None:
                    m1.bias.data = m0.bias.data.clone()
                m1.running_mean = m0.running_mean.clone()
                m1.running_var = m0.running_var.clone()

        else:
            if isinstance(m0, nn.BatchNorm2d):
                idx1 = np.squeeze(np.argwhere(np.asarray(bn_mask[bn_i].cpu().numpy())))
                if idx1.size == 1:
                    idx1 = np.resize(idx1, (1,))
                m1.weight.data = m0.weight.data[idx1].clone()
                if m1.bias is not None:
                    m1.bias.data = m0.bias.data[idx1].clone()
                m1.running_mean = m0.running_mean[idx1].clone()
                m1.running_var = m0.running_var[idx1].clone()
                bn_i += 1
            elif isinstance(m0, nn.Conv2d):
                if (isinstance(conv_in_mask[conv_i], list)):
                    idx0 = np.squeeze(np.argwhere(np.asarray(torch.cat(conv_in_mask[conv_i], 0).cpu().numpy())))
                else:
                    idx0 = np.squeeze(np.argwhere(np.asarray(conv_in_mask[conv_i].cpu().numpy())))
                idx1 = np.squeeze(np.argwhere(np.asarray(conv_out_mask[conv_i].cpu().numpy())))
                if idx0.size == 1:
                    idx0 = np.resize(idx0, (1,))
                if idx1.size == 1:
                    idx1 = np.resize(idx1, (1,))
                w = m0.weight.data[:, idx0, :, :].clone()
                m1.weight.data = w[idx1, :, :, :].clone()
                if m1.bias is not None:
                    m1.bias.data = m0.bias.data[idx1].clone()
                conv_i += 1

        scale_i += 1

    return new_model


def load_prune_model(model, pruned_model_dict_path):
    _load = torch.load(pruned_model_dict_path)
    prued_mask = _load['prued_mask']
    bn_index = _load['bn_index']
    prune_model = get_new_model(model, model, prued_mask, bn_index)
    return prune_model



