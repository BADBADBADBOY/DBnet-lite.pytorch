"""
#!-*- coding=utf-8 -*-
@author: BADBADBADBADBOY
@contact: 2441124901@qq.com
@software: PyCharm Community Edition
@file: dataload.py
@time: 2020/7/4 15:16

"""
import numpy as np
from PIL import Image
from torch.utils import data
import glob
import cv2
import random
import os
import torchvision.transforms as transforms
import torch
from .random_thansform import Random_Augment
from .MakeSegMap import MakeSegDetectionData
from .MakeBorderMap import MakeBorderMap

def get_img(img_path):
    img = cv2.imread(img_path)
    return img

def get_bboxes(gt_path,config):
    with open(gt_path,'r',encoding='utf-8') as fid:
        lines = fid.readlines()
    polys = []
    tags = []
    for line in lines:
        line = line.replace('\ufeff','').replace( '\xef\xbb\xbf','')
        gt = line.split(',')
        if "#" in gt[-1]:
            tags.append(True)
        else:
            tags.append(False)
        if(config['train']['is_icdar2015']):
            box = [int(gt[i]) for i in range(8)]
        else:
            box = [int(gt[i]) for i in range(len(gt)-1)]
        polys.append(box)
    return np.array(polys), tags

class DataLoader(data.Dataset):
    def __init__(self, config):
        self.config = config
        self.ra = Random_Augment()
        self.ms = MakeSegDetectionData()
        self.mb = MakeBorderMap()
        img_paths = glob.glob(os.path.join(config['train']['train_img_dir'],'*'+config['train']['train_img_format']))
        gt_paths = []
        for img_path in img_paths:
            im_name = img_path.split('/')[-1].split('.')[0]
            if(config['train']['is_icdar2015']):
                gt_file_name = 'gt_'+im_name+'.txt'
            else:
                gt_file_name = im_name + '.txt'
            gt_paths.append(os.path.join(config['train']['train_gt_dir'],gt_file_name))
        self.img_paths = img_paths
        self.gt_paths = gt_paths

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        gt_path = self.gt_paths[index]

        img = get_img(img_path)
        polys, dontcare = get_bboxes(gt_path,self.config)

        if self.config['train']['is_transform']:
            img, polys = self.ra.random_scale(img, polys, 640)
            img, polys = self.ra.random_rotate(img, polys,self.config['train']['radom_angle'])
            img, polys = self.ra.random_flip(img, polys)
            img, polys, dontcare = self.ra.random_crop_db(img, polys, dontcare)

        img, gt, gt_mask = self.ms.process(img, polys, dontcare)
        img, thresh_map, thresh_mask = self.mb.process(img, polys, dontcare)

        if self.config['train']['is_show']:
            cv2.imwrite('img.jpg',img)
            cv2.imwrite('gt.jpg',gt[0]*255)
            cv2.imwrite('gt_mask.jpg',gt_mask[0]*255)
            cv2.imwrite('thresh_map.jpg',thresh_map*255)
            cv2.imwrite('thresh_mask.jpg',thresh_mask*255)

        if self.config['train']['is_transform']:
            img = Image.fromarray(img)
            img = img.convert('RGB')
            img = transforms.ColorJitter(brightness=32.0 / 255, saturation=0.5)(img)
        else:
            img = Image.fromarray(img)
            img = img.convert('RGB')

        img = transforms.ToTensor()(img)
        img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)

        gt = torch.from_numpy(gt).float()
        gt_mask = torch.from_numpy(gt_mask).float()
        thresh_map = torch.from_numpy(thresh_map).float()
        thresh_mask = torch.from_numpy(thresh_mask).float()

        return img, gt,gt_mask,thresh_map,thresh_mask







