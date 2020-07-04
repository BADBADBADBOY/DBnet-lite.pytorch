"""
#!-*- coding=utf-8 -*-
@author: BADBADBADBADBOY
@contact: 2441124901@qq.com
@software: PyCharm Community Edition
@file: prune.py
@time: 2020/6/27 10:23

"""
import sys
sys.path.append('/home/aistudio/external-libraries')
from models.DBNet import DBNet
import torch
import torch.nn as  nn
import numpy as np
import collections
import torchvision.transforms as transforms
import cv2
import os
import argparse
import math
from PIL import Image
from torch.autograd import Variable

def resize_image(img,short_side=736):
    height, width, _ = img.shape
    if height < width:
        new_height = short_side
        new_width = int(math.ceil(new_height / height * width / 32) * 32)
    else:
        new_width = short_side
        new_height = int(math.ceil(new_width / width * height / 32) * 32)
    resized_img = cv2.resize(img, (new_width, new_height))
    return resized_img

def prune(args):


    img = cv2.imread(args.img_file)
    img = resize_image(img)
    img = Image.fromarray(img)
    img = img.convert('RGB')
    img = transforms.ToTensor()(img)
    img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
    img = Variable(img.cuda()).unsqueeze(0)

    model = DBNet(args.backbone, adaptive=False).cuda()
    model_dict = torch.load(args.checkpoint)['state_dict']
    state = model.state_dict()
    for key in state.keys():
        if key in model_dict.keys():
            state[key] = model_dict[key]
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        out = model(img)
    cv2.imwrite('re.jpg',out[0,0].cpu().numpy()*255)


    bn_weights = []
    for m in model.modules():
        if (isinstance(m, nn.BatchNorm2d)):
            bn_weights.append(m.weight.data.abs().clone())
    bn_weights = torch.cat(bn_weights, 0)

    sort_result, sort_index = torch.sort(bn_weights)

    thresh_index = int(args.cut_percent * bn_weights.shape[0])

    if (thresh_index == bn_weights.shape[0]):
        thresh_index = bn_weights.shape[0] - 1

    prued = 0
    prued_mask = []
    bn_index = []
    conv_index = []
    remain_channel_nums = []
    for k, m in enumerate(model.modules()):
        if (isinstance(m, nn.BatchNorm2d)):
            bn_weight = m.weight.data.clone()
            mask = bn_weight.abs().gt(sort_result[thresh_index])
            remain_channel = mask.sum()

            if (remain_channel == 0):
                remain_channel = 1
                mask[int(torch.argmax(bn_weight))] = 1

            v = 0
            n = 1
            if (remain_channel % args.base_num != 0):
                if (remain_channel > args.base_num):
                    while (v < remain_channel):
                        n += 1
                        v = args.base_num * n
                    if (remain_channel - (v - args.base_num) < v - remain_channel):
                        remain_channel = v - args.base_num
                    else:
                        remain_channel = v
                    if (remain_channel > bn_weight.size()[0]):
                        remain_channel = bn_weight.size()[0]
                    remain_channel = torch.tensor(remain_channel)
                    result, index = torch.sort(bn_weight)
                    mask = bn_weight.abs().ge(result[-remain_channel])

            remain_channel_nums.append(int(mask.sum()))
            prued_mask.append(mask)
            bn_index.append(k)
            prued += mask.shape[0] - mask.sum()
        elif (isinstance(m, nn.Conv2d)):
            conv_index.append(k)
    print(remain_channel_nums)
    print('total_prune_ratio:', float(prued) / bn_weights.shape[0])
    print(bn_index)

    new_model = DBNet(args.backbone, adaptive=False).cuda()

    merge1_index = [13, 17, 24, 32]
    merge2_index = [41, 45, 52, 60, 68]
    merge3_index = [77, 81, 88, 96, 104, 112, 120]
    merge4_index = [129, 133, 140, 148]

    index_0 = []
    for item in merge1_index:
        index_0.append(bn_index.index(item))
    mask1 = prued_mask[index_0[0]] | prued_mask[index_0[1]] | prued_mask[index_0[2]] | prued_mask[index_0[3]]

    index_1 = []
    for item in merge2_index:
        index_1.append(bn_index.index(item))
    mask2 = prued_mask[index_1[0]] | prued_mask[index_1[1]] | prued_mask[index_1[2]] | prued_mask[index_1[3]] | prued_mask[
        index_1[4]]

    index_2 = []
    for item in merge3_index:
        index_2.append(bn_index.index(item))
    mask3 = prued_mask[index_2[0]] | prued_mask[index_2[1]] | prued_mask[index_2[2]] | prued_mask[index_2[3]] | prued_mask[
        index_2[4]] | prued_mask[index_2[5]] | prued_mask[index_2[6]]

    index_3 = []
    for item in merge4_index:
        index_3.append(bn_index.index(item))
    mask4 = prued_mask[index_3[0]] | prued_mask[index_3[1]] | prued_mask[index_3[2]] | prued_mask[index_3[3]]


    for index in index_0:
        prued_mask[index] = mask1

    for index in index_1:
        prued_mask[index] = mask2

    for index in index_2:
        prued_mask[index] = mask3

    for index in index_3:
        prued_mask[index] = mask4

    print(new_model)
##############################################################
    index_bn = 0
    index_conv = 0

    bn_mask = []
    conv_in_mask = []
    conv_out_mask = []

    for m in new_model.modules():
        if (isinstance(m, nn.BatchNorm2d)):
            m.num_features = prued_mask[index_bn].sum()
            bn_mask.append(prued_mask[index_bn])
            index_bn += 1
        elif (isinstance(m, nn.Conv2d)):
            if(index_conv == 0):
                m.in_channels = 3
                conv_in_mask.append(torch.ones(3))
            else:
                m.in_channels = prued_mask[index_conv - 1].sum()
                conv_in_mask.append(prued_mask[index_conv - 1])
            m.out_channels = prued_mask[index_conv].sum()
            conv_out_mask.append(prued_mask[index_conv])
            index_conv += 1
        if (index_bn > len(bn_index) - 3):
            break

    conv_change_index = [16,44,80,132]  # 
    change_conv_bn_index = [3,32,68,120] # 
    tag = 0
    for m in new_model.modules():
        if (isinstance(m, nn.Conv2d)):
            if(tag in conv_change_index):
                index = conv_change_index.index(tag)
                index = change_conv_bn_index[index]
                index =bn_index.index(index)
                mask = prued_mask[index]
                conv_in_mask[index+4] = mask
                m.in_channels = mask.sum()
        tag+=1

 

    bn_i = 0
    conv_i = 0
    scale_i = 0 
    scale_mask = [mask4,mask3,mask2,mask1]
    for [m0, m1] in zip(model.modules(), new_model.modules()):
        if (bn_i > len(bn_mask)-1):
            if isinstance(m0, nn.Conv2d):
                # import pdb
                # pdb.set_trace()
                if(scale_i<4):
                    m1.in_channels = scale_mask[scale_i].sum()
                    idx0 = np.squeeze(np.argwhere(np.asarray(scale_mask[scale_i].cpu().numpy())))
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
                scale_i+=1

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

    print(new_model)
    new_model.eval()
    with torch.no_grad():
        out = new_model(img)
    print(out.shape)
    cv2.imwrite('re1.jpg',out[0,0].cpu().numpy()*255)

    save_obj = {'prued_mask': prued_mask, 'bn_index': bn_index, 'state_dict': new_model.state_dict()}
    torch.save(save_obj, os.path.join(args.save_prune_model_path, 'pruned_dict.pth.tar'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--backbone', nargs='?', type=str, default='resnet50')

    parser.add_argument('--num_workers', nargs='?', type=int, default=0,
                        help='num workers to train')
    parser.add_argument('--base_num', nargs='?', type=int, default=8,
                        help='Base after Model Channel Clipping')
    parser.add_argument('--cut_percent', nargs='?', type=float, default=0.9,
                        help='Model channel clipping scale')
    parser.add_argument('--checkpoint', default='./checkpoints/DB_resnet50_bs_16_ep_1200/DB.pth.tar',
                        type=str, metavar='PATH',
                        help='ori model path')
    parser.add_argument('--save_prune_model_path', default='./pruned/checkpoints/', type=str, metavar='PATH',
                        help='pruned model path')
    parser.add_argument('--img_file',
                        default='/home/aistudio/work/data/icdar/test_img/img_10.jpg',
                        type=str,
                        help='')
    args = parser.parse_args()

    prune(args)
