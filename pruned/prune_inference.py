# -*- coding:utf-8 _*-
"""
@author:fxw
@file: test.py
@time: 2020/04/28
"""

import sys

sys.path.append('/home/aistudio/external-libraries')
sys.path.append('./')
import cv2
import torch
import math
import os
import glob
import time
import copy
import yaml
import argparse
import pyclipper
import numpy as np
from tqdm import tqdm
from models.DBNet import DBNet
import torchvision.transforms as transforms
from shapely.geometry import Polygon
from cal_rescall.script import cal_recall_precison_f1
from utils.DB_postprocesss import *
from utils.tools import *
from utils.fuse_model import fuse_module
from pruned.get_pruned_model import load_prune_model


def test_net(config):
    os.environ["CUDA_VISIBLE_DEVICES"] = config['test']['gpu_id']

    config['train']['pretrained'] = config['test']['pretrained']
    files = glob.glob(os.path.join(config['test']['test_img_dir'], '*' + config['test']['test_img_format']))
    model = DBNet(config)
    model = load_prune_model(model, config['pruned']['checkpoints_dict']).cuda()

    model_dict = torch.load(config['pruned']['checkpoints'])['state_dict']
    state = model.state_dict()
    for key in state.keys():
        if key in model_dict.keys():
            state[key] = model_dict[key]
    model.load_state_dict(state)

    if (config['test']['merge_conv_bn']):
        fuse_module(model)
        print('merge conv bn ok!!!')

    if not (os.path.exists(config['test']['out_dir'])):
        os.mkdir(config['test']['out_dir'])

    if not (os.path.exists(os.path.join(config['test']['out_dir'], 'img_text'))):
        os.mkdir(os.path.join(config['test']['out_dir'], 'img_text'))

    if not (os.path.exists(os.path.join(config['test']['out_dir'], 'img_result'))):
        os.mkdir(os.path.join(config['test']['out_dir'], 'img_result'))

    bar = tqdm(total=len(files))
    params = {'thresh': config['test']['thresh'],
              'box_thresh': config['test']['box_thresh'],
              'max_candidates': config['test']['max_candidates'],
              'is_poly': config['test']['is_poly'],
              'unclip_ratio': config['test']['unclip_ratio'],
              'min_size': config['test']['min_size']
              }

    dbprocess = DBPostProcess(params)

    total_frame = 0.0
    total_time = 0.0

    for file in files:
        model.eval()
        bar.update(1)
        img = cv2.imread(file)
        img_ori = img.copy()
        img_name = file.split('/')[-1].split('.')[0]
        img = resize_image(img, config['test']['short_side'])

        img = transforms.ToTensor()(img)
        img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img).unsqueeze(0).cuda()

        torch.cuda.synchronize()
        start = time.time()

        with torch.no_grad():
            out = model(img)

        scale = (img_ori.shape[1] * 1.0 / out.shape[3], img_ori.shape[0] * 1.0 / out.shape[2])
        bbox_batch, score_batch = dbprocess(out.cpu().numpy(), [scale])

        torch.cuda.synchronize()
        end = time.time()
        total_frame += 1
        total_time += (end - start)
        sys.stdout.flush()

        for bbox in bbox_batch[0]:
            bbox = bbox.reshape(-1, 2).astype(np.int)
            # bbox = sort_coord(bbox)
            img_ori = cv2.drawContours(img_ori.copy(), [bbox], -1, (0, 255, 0), 1)

        if config['test']['is_icdar2015']:
            text_file = 'res_' + img_name + '.txt'
        else:
            text_file = img_name + '.txt'

        with open(os.path.join(config['test']['out_dir'], 'img_text', text_file), 'w+', encoding='utf-8') as fid:
            for bbox in bbox_batch[0]:
                if (len(bbox) == 0):
                    continue
                bbox = bbox.reshape(-1, 2).astype(np.int)
                # bbox = sort_coord(bbox)
                bbox = bbox.reshape(-1).tolist()
                bbox = [str(x) for x in bbox]
                bbox = ','.join(bbox)
                fid.write(bbox + '\n')

        cv2.imwrite(os.path.join(config['test']['out_dir'], 'img_result', img_name + '.jpg'), img_ori)
    bar.close()
    print('fps: %.2f' % (total_frame / total_time))
    result_dict = cal_recall_precison_f1(config['test']['test_gt_dir'],
                                         os.path.join(config['test']['out_dir'], 'img_text'))
    return result_dict


if __name__ == '__main__':
    stream = open('./config.yaml', 'r', encoding='utf-8')
    config = yaml.load(stream, Loader=yaml.FullLoader)
    result_dict = test_net(config)
    print(result_dict)