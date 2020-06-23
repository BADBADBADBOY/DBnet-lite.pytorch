# -*- coding:utf-8 _*-
"""
@author:fxw
@file: test.py
@time: 2020/04/28
"""

import sys
sys.path.append('/home/aistudio/external-libraries')
import cv2
import torch
import os
import glob
import argparse
import pyclipper
from models.DBNet import DBNet
import torchvision.transforms as transforms
import numpy as np
from shapely.geometry import Polygon
from tqdm import tqdm
import time
from cal_rescall.script import cal_recall_precison_f1
from utils.DB_postprocesss import *


def scale_img(img, long_size=1120):
    h, w = img.shape[0:2]
    if (h > w):
        scale = long_size * 1.0 / h
        w = w * scale
        w = int((w // 32 + 1) * 32)
        img = cv2.resize(img, dsize=(w, long_size))
    else:
        scale = long_size * 1.0 / w
        h = scale * h
        h = int((h // 32 + 1) * 32)
        img = cv2.resize(img, dsize=(long_size, h))
    return img


def test_net(args):
    files = glob.glob(os.path.join(args.test_dir, '*.jpg'))
    model = DBNet(args.model_name, adaptive=False).cuda()

    model_dict = torch.load(args.checkpoint)['state_dict']
    state = model.state_dict()
    for key in state.keys():
        if key in model_dict.keys():
            state[key] = model_dict[key]
    model.load_state_dict(state)

    if not (os.path.exists(args.out_dir)):
        os.mkdir(args.out_dir)

    if not (os.path.exists(os.path.join(args.out_dir, 'img_text'))):
        os.mkdir(os.path.join(args.out_dir, 'img_text'))

    if not (os.path.exists(os.path.join(args.out_dir, 'img_result'))):
        os.mkdir(os.path.join(args.out_dir, 'img_result'))

    bar = tqdm(total=len(files))

    params = {'thresh': args.thresh, 'box_thresh': args.box_thresh, 'max_candidates': args.max_candidates}
    dbprocess = DBPostProcess(params)

    for file in files:
        model.eval()
        bar.update(1)
        start = time.time()
        img = cv2.imread(file)
        img_ori = img.copy()
        img_name = file.split('/')[-1].split('.')[0]
        #         img =  scale_img(img, long_size=1360)
        #         img = cv2.resize(img,((img.shape[1]//32+1)*32,(img.shape[0]//32+1)*32))
        img = cv2.resize(img, (1280, 736))
        img_sh = img.copy()

        img = transforms.ToTensor()(img)
        img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img).unsqueeze(0).cuda()

        with torch.no_grad():
            out = model(img)

        scale = (img_ori.shape[1] * 1.0 / out.shape[3], img_ori.shape[0] * 1.0 / out.shape[2])
        new_bbox = dbprocess(out.cpu().numpy(), [scale])
        for bbox in new_bbox[0]:
            img_ori = cv2.drawContours(img_ori.copy(), [bbox.reshape(4, 2).astype(np.int)], -1, (0, 255, 0), 1)

        with open(os.path.join(args.out_dir, 'img_text', 'res_' + img_name + '.txt'), 'w+', encoding='utf-8') as fid:
            for bbox in new_bbox[0]:
                if (len(bbox) == 0):
                    continue
                bbox = bbox.reshape(-1).tolist()
                bbox = [str(x) for x in bbox]
                bbox = ','.join(bbox)
                fid.write(bbox + '\n')

        cv2.imwrite(os.path.join(args.out_dir, 'img_result', img_name + '.jpg'), img_ori)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--min_area', nargs='?', type=float, default=50,
                        help='min area')
    parser.add_argument('--model_name', nargs='?', type=str,
                        default='resnet18')  # deformable_resnet18,resnet18,resnet50,deformable_resnet50
    parser.add_argument('--test_dir', nargs='?', type=str,
                        default='/src/notebooks/fxw20190611/PSENet-master/train_data/icdar2015/test/image/',
                        help='test dir')
    parser.add_argument('--gt_dir', nargs='?', type=str,
                        default='/src/notebooks/fxw20190611/PSENet-master/train_data/icdar2015/test/label/',
                        help='gt dir')
    parser.add_argument('--box_thresh', nargs='?', type=float, default=0.6,
                        help='box_thresh')
    parser.add_argument('--max_candidates', nargs='?', type=int, default=1000,
                        help='max_candidates')
    parser.add_argument('--thresh', nargs='?', type=float, default=0.5,
                        help='thresh')
    parser.add_argument('--checkpoint', default='./checkpoints/model.pth', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--out_dir', default='./output', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    args = parser.parse_args()

    test_net(args)
    from cal_rescall.script import cal_recall_precison_f1
    result_dict = cal_recall_precison_f1(args.gt_dir, os.path.join(args.out_dir, 'img_text'))
    print(result_dict)