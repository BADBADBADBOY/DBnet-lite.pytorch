"""
#!-*- coding=utf-8 -*-
@author: BADBADBADBADBOY
@contact: 2441124901@qq.com
@software: PyCharm Community Edition
@file: train.py
@time: 2020/7/4 15:16

"""
import sys
sys.path.append('/home/aistudio/external-libraries')
import os
import torch
import yaml
import argparse
import numpy as np
olderr = np.seterr(all='ignore')
from models.DBNet import DBNet
from torch.autograd import Variable
from loss.loss import L1BalanceCELoss
from dataloader.dataload import DataLoader
from utils.Logger import Logger
from utils.metrics import runningScore
from utils.model_eval import val
from utils.tools import *
from utils.set_optimizer import *

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def set_seed(seed):
    import numpy as np
    import random
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

GLOBAL_WORKER_ID = None
GLOBAL_SEED = 2020

def worker_init_fn(worker_id):
    global GLOBAL_WORKER_ID
    GLOBAL_WORKER_ID = worker_id
    set_seed(GLOBAL_SEED + worker_id)

def train_net(config):
    os.environ["CUDA_VISIBLE_DEVICES"] = config['train']['gpu_id']
    data_loader = DataLoader(config)
    train_loader = torch.utils.data.DataLoader(
        data_loader,
        batch_size=config['train']['batch_size'],
        shuffle=True,
        num_workers=config['train']['num_workers'],
        worker_init_fn = worker_init_fn,
        drop_last=True,
        pin_memory=False)

    start_epoch = 0
    running_metric_binary = runningScore(2)

    if not (os.path.exists(config['train']['checkpoints'])):
            os.mkdir(config['train']['checkpoints'])
    checkpoints = os.path.join(config['train']['checkpoints'],"DB_%s_bs_%d_ep_%d" % (config['train']['backbone'],
                          config['train']['batch_size'], config['train']['n_epoch']))
    if not (os.path.exists(checkpoints)):
            os.mkdir(checkpoints)

    
    model = DBNet(config).cuda()
    criterion = L1BalanceCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=config['train']['base_lr'], momentum=0.99, weight_decay=5e-4)

    if config['train']['restore']:
        print('Resuming from checkpoint.')
        assert os.path.isfile(config['train']['resume']), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(config['train']['resume'])
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        log_write = Logger(os.path.join(checkpoints, 'log.txt'), title=config['train']['backbone'], resume=True)
    else:
        print('Training from scratch.')
        log_write = Logger(os.path.join(checkpoints,'log.txt'), title=config['train']['backbone'])
        log_write.set_names(['   epoch', 'Total loss', '  Bce loss', 'Thresh loss', '  L1 loss', 'Binary Acc', 'Binary IoU', '   rescall',' precision','   hmean'])
    max_hmean = -1
    for epoch in range(start_epoch,config['train']['n_epoch']):
        model.train()

        bce_loss_list = []
        thresh_loss_list = []
        l1_loss_list = []
        total_loss_list = []

        if(config['train']['decay_method']=='e_decay'):
            adjust_learning_rate_poly(config['train']['base_lr'], optimizer, epoch, max_epoch=config['train']['n_epoch'], factor=0.9)
        else:
            adjust_learning_rate(config, optimizer, epoch,config['train']['gama'])

        for batch_idx, (imgs, gts, gt_masks, thresh_maps, thresh_masks) in enumerate(train_loader):
            imgs = Variable(imgs.cuda())
            gts = Variable(gts.cuda())
            gt_masks = Variable(gt_masks.cuda())
            thresh_maps = Variable(thresh_maps.cuda())
            thresh_masks = Variable(thresh_masks.cuda())
            batch = {}
            batch['gt'] = gts
            batch['mask'] = gt_masks
            batch['thresh_map'] = thresh_maps
            batch['thresh_mask'] = thresh_masks

            pre = model(imgs)
            loss, metrics = criterion(pre, batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            score_binary = cal_binary_score(pre['binary'], gts, gt_masks.unsqueeze(1), running_metric_binary)

            bce_loss_list.append(metrics['bce_loss'].item())
            thresh_loss_list.append(metrics['thresh_loss'].item())
            l1_loss_list.append(metrics['l1_loss'].item())
            total_loss_list.append(loss.item())
            if batch_idx % config['train']['show_step'] == 0:
                if(config['train']['print_format']=='linux'):
                    headers = ['epoch/epochs','batch/batchs' ,'TotalLoss' ,'BceLoss',' ThreshLoss','L1Loss', 'Binary Acc','Binary IoU', 'Lr Rate']
                    show_item = [[str(epoch)+'/'+str(config['train']['n_epoch']),
                                    str(batch_idx + 1)+'/'+str(len(train_loader)),
                                    get_str(np.mean(total_loss_list)),
                                    get_str(np.mean(bce_loss_list)),
                                    get_str(np.mean(thresh_loss_list)),
                                    get_str(np.mean(l1_loss_list)),
                                    get_str(score_binary['Mean Acc']),
                                    get_str(score_binary['Mean IoU']),
                                    get_str(optimizer.param_groups[0]['lr'])
                                ]]
                    print_table(headers,show_item,type_str='train')
                else:
                    output_log = '({epoch}/{epochs}/{batch}/{size}) | TotalLoss: {total_loss:.4f} | BceLoss: {bce_loss:.4f} | ThreshLoss: {thresh_loss: .4f} | L1Loss: {l1_loss: .4f} | Binary Acc: {bin_acc: .4f} | Binary IoU: {bin_iou: .4f} | Lr: {lr: .4f}'.format(
                    epoch=epoch,
                    epochs=config['train']['n_epoch'] ,
                    batch=batch_idx + 1,
                    size=len(train_loader),
                    total_loss=np.mean(total_loss_list),
                    bce_loss=np.mean(bce_loss_list),
                    thresh_loss=np.mean(thresh_loss_list),
                    l1_loss=np.mean(l1_loss_list),
                    bin_acc=score_binary['Mean Acc'],
                    bin_iou=score_binary['Mean IoU'],
                    lr=optimizer.param_groups[0]['lr']
                    )
                    print(output_log)
                    sys.stdout.flush()
        
        if( epoch > config['train']['start_val_epoch']):
            result_dict = val(model,config)
            rescall,precision,hmean = result_dict['recall'],result_dict['precision'],result_dict['hmean']
            print('epoch:',epoch,'rescall:',rescall,'precision:',precision,'hmean:',hmean)
        else:
            rescall = 0
            precision = 0
            hmean = 0   
        log_write.append([epoch, np.mean(total_loss_list), np.mean(bce_loss_list), np.mean(thresh_loss_list),
                            np.mean(l1_loss_list), score_binary['Mean Acc'], score_binary['Mean IoU'],
                            rescall,precision,hmean])
        if(hmean > max_hmean and config['train']['start_val_epoch'] < config['train']['n_epoch']):
            max_hmean = hmean
            save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'lr': config['train']['base_lr'],
            'optimizer': optimizer.state_dict(),
        }, checkpoint=checkpoints,filename='best_model.pth.tar')

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'lr': config['train']['base_lr'],
            'optimizer': optimizer.state_dict(),
        }, checkpoint=checkpoints)
        


if __name__ == '__main__':
    stream = open('config.yaml', 'r', encoding='utf-8')
    config = yaml.load(stream,Loader=yaml.FullLoader)
    train_net(config)