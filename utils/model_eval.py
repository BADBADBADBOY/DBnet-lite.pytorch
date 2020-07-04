"""
#!-*- coding=utf-8 -*-
@author: BADBADBADBADBOY
@contact: 2441124901@qq.com
@software: PyCharm Community Edition
@file: model_eval.py
@time: 2020/7/4 15:16

"""
import sys
import cv2
import torch
import math
import os
import glob
import argparse
import pyclipper
import torchvision.transforms as transforms
import numpy as np
from shapely.geometry import Polygon
from tqdm import tqdm
import time
from cal_rescall.script import cal_recall_precison_f1
from .DB_postprocesss import DBPostProcess
import copy
import yaml
from utils.tools import *
from PIL import Image

def val(model,config):
    model.eval()
    files = glob.glob(os.path.join(config['train']['val_img_dir'],'*'+config['train']['val_img_format']))
    if not (os.path.exists(config['train']['output_path'])):
        os.mkdir(config['train']['output_path'])
    
    if not (os.path.exists(os.path.join(config['train']['output_path'],'img_text'))):
        os.mkdir(os.path.join(config['train']['output_path'],'img_text'))
        
    if not (os.path.exists(os.path.join(config['train']['output_path'],'img_result'))):
        os.mkdir(os.path.join(config['train']['output_path'],'img_result'))
    
    bar = tqdm(total=len(files))
    
    
    params = {'thresh':config['test']['thresh'],
              'box_thresh':config['test']['box_thresh'],
              'max_candidates':config['test']['max_candidates'],
              'is_poly':config['test']['is_poly'],
              'unclip_ratio':config['test']['unclip_ratio'],
              'min_size':config['test']['min_size']
              }

    dbprocess = DBPostProcess(params)
    total_frame = 0.0
    total_time = 0.0
    for file in files:
        
        bar.update(1)
        img = cv2.imread(file)
        img_ori = img.copy()
        img_name = file.split('/')[-1].split('.')[0]
        img = resize_image(img,config['test']['short_side'])
        img = Image.fromarray(img)
        img = img.convert('RGB')
        img = transforms.ToTensor()(img)
        img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img).unsqueeze(0).cuda()

        torch.cuda.synchronize()
        start = time.time()
        
        with torch.no_grad():
            out = model(img)

        scale = (img_ori.shape[1] * 1.0 / out.shape[3], img_ori.shape[0] * 1.0 / out.shape[2])       
        bbox_batch,score_batch = dbprocess(out.cpu().numpy(),[scale])

        torch.cuda.synchronize()
        end = time.time()
        total_frame += 1
        total_time += (end - start)
        sys.stdout.flush()

        for bbox in bbox_batch[0]:
            img_ori = cv2.drawContours(img_ori.copy(), [bbox.reshape(-1, 2).astype(np.int)], -1, (0, 255, 0), 1)

        if config['test']['is_icdar2015']:
            text_file = 'res_' + img_name + '.txt'
        else:
            text_file = img_name + '.txt'

        with open(os.path.join(config['train']['output_path'],'img_text',text_file),'w+',encoding='utf-8') as fid:
            for bbox in bbox_batch[0]:
                if(len(bbox)==0):
                    continue
                bbox = bbox.reshape(-1).tolist()
                bbox = [str(x) for x in bbox]
                bbox = ','.join(bbox)
                fid.write(bbox+'\n')
                
        cv2.imwrite(os.path.join(config['train']['output_path'],'img_result',img_name+'.jpg'),img_ori)
    bar.close()
    print('fps: %.2f'%(total_frame / total_time))
    from cal_rescall.script import cal_recall_precison_f1
    result_dict = cal_recall_precison_f1(config['train']['val_gt_dir'], os.path.join(config['train']['output_path'], 'img_text'))
    return result_dict