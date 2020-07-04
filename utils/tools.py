"""
#!-*- coding=utf-8 -*-
@author: BADBADBADBADBOY
@contact: 2441124901@qq.com
@software: PyCharm Community Edition
@file: tools.py
@time: 2020/7/4 15:16

"""

import copy
import math
import cv2
import os
import torch
import numpy as np
from tabulate import tabulate

def judgePoint(point1,point2,center_point):
    if(point1[0]<=0 and point2[0]>0):
        return True
    if(point1[0]==0 and point2[0]==0):
        return point1[1]<point2[1]
    det = (point1[0]-center_point[0])*(point2[1]-center_point[1])-(point2[0]-center_point[0])*(point1[1]-center_point[1])
    if(det>0):
        return True
    if(det<0):
        return False
    d1 = (point1[0]-center_point[0])*(point1[0]-center_point[0])-(point1[1]-center_point[1])*(point1[1]-center_point[1])
    d2 = (point2[0]-center_point[0])*(point2[0]-center_point[0])-(point2[1]-center_point[1])*(point2[1]-center_point[1])
    return d1<d2

def sort_coord(coords):
    x = 0
    y = 0
    for i in range(coords.shape[0]):
        x+=coords[i][0]
        y+=coords[i][1]
    center_x = x/coords.shape[0]
    center_y = y/coords.shape[0]

    for i in range(coords.shape[0]):
        for j in range(coords.shape[0]-i-1):
            if(judgePoint(coords[j],coords[j+1],(center_x,center_y))):
                tmp = copy.deepcopy(coords[j])
                coords[j] = copy.deepcopy(coords[j+1])
                coords[j+1] = copy.deepcopy(tmp)

    return coords
    
def print_table(header,item,type_str):
    os.system('clear')
    print(type_str+'....')
    print (tabulate(item, header, tablefmt="grid"))
        
def get_str(_str,num=5):
    return str("%.{}f".format(num) % _str)
    
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

def cal_binary_score(binarys, gt_binarys, training_masks, running_metric_binary, thresh=0.5):
    training_masks = training_masks.data.cpu().numpy()
    pred_binary = binarys.data.cpu().numpy() * training_masks
    pred_binary[pred_binary <= thresh] = 0
    pred_binary[pred_binary > thresh] = 1
    pred_binary = pred_binary.astype(np.int32)
    gt_binary = gt_binarys.data.cpu().numpy() * training_masks
    gt_binary = gt_binary.astype(np.int32)
    running_metric_binary.update(gt_binary, pred_binary)
    score_binary, _ = running_metric_binary.get_scores()
    return score_binary


def save_checkpoint(state, checkpoint='checkpoints', filename='DB.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)