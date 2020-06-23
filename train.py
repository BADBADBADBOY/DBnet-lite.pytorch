# -*- coding:utf-8 _*-
"""
@author:fxw
@file: train.py.py
@time: 2020/04/28
"""
import sys
sys.path.append('/home/aistudio/external-libraries')
import os
import torch
import argparse
from models.DBNet import DBNet
from torch.autograd import Variable
from loss.loss import L1BalanceCELoss
from dataloader.dataload import DataLoader
import numpy as np
from utils.Logger import Logger
from utils.metrics import runningScore


def lr_poly(base_lr, epoch, max_epoch=1200, factor=0.9):
    return base_lr*((1-float(epoch)/max_epoch)**(factor))


def adjust_learning_rate_poly(base_lr, optimizer, epoch, max_epoch=1200, factor=0.9):
    lr = lr_poly(base_lr, epoch, max_epoch, factor)
    optimizer.param_groups[0]['lr'] = lr


def adjust_learning_rate(args, optimizer, epoch):
    global state
    if epoch in args.schedule:
        args.lr = args.lr * 0.1
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr


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

def train_net(args):
    data_loader = DataLoader(is_transform=True)
    train_loader = torch.utils.data.DataLoader(
        data_loader,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_worker,
        drop_last=True,
        pin_memory=True)

    
    start_epoch = 0
    running_metric_binary = runningScore(2)

    if args.checkpoints == '':
        args.checkpoints = "checkpoints/DB_%s_bs_%d_ep_%d" % (args.model_name, args.batch_size, args.n_epoch)

    if not (os.path.exists(args.checkpoints)):
            os.mkdir(args.checkpoints)

    
    model = DBNet(args.model_name, adaptive=True).cuda()
    criterion = L1BalanceCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.99, weight_decay=5e-4)
    if args.resume:
        print('Resuming from checkpoint.')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        log_write = Logger(os.path.join(args.checkpoints, 'log.txt'), title=args.model_name, resume=True)
    else:
        print('Training from scratch.')
        log_write = Logger(os.path.join(args.checkpoints,'log.txt'), title=args.model_name)
        log_write.set_names(['    epoch', 'Total loss', '  Bce loss', 'Thresh loss', '  L1 loss', 'Binary Acc', 'Binary IoU', 'Learn rate'])

    model.train()

    for epoch in range(start_epoch,args.n_epoch):

        bce_loss_list = []
        thresh_loss_list = []
        l1_loss_list = []
        total_loss_list = []

        # adjust_learning_rate(args, optimizer, epoch)
        adjust_learning_rate_poly(args.lr, optimizer, epoch, max_epoch=args.n_epoch, factor=0.9)

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
            if batch_idx % args.show == 0:
                output_log = '({epoch}/{epochs}/{batch}/{size}) | TotalLoss: {total_loss:.4f} | BceLoss: {bce_loss:.4f} | ThreshLoss: {thresh_loss: .4f} | L1Loss: {l1_loss: .4f} | Binary Acc: {bin_acc: .4f} | Binary IoU: {bin_iou: .4f} | Lr: {lr: .4f}'.format(
                    epoch=epoch,
                    epochs=args.n_epoch,
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
        log_write.append([epoch, np.mean(total_loss_list), np.mean(bce_loss_list), np.mean(thresh_loss_list),
                            np.mean(l1_loss_list), score_binary['Mean Acc'], score_binary['Mean IoU'],
                            optimizer.param_groups[0]['lr']])
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'lr': args.lr,
            'optimizer': optimizer.state_dict(),
        }, checkpoint=args.checkpoints)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--model_name', nargs='?', type=str,
                        default='deformable_resnet18')  # deformable_resnet18,resnet18,resnet50,deformable_resnet50
    parser.add_argument('--n_epoch', nargs='?', type=int, default=1200,
                        help='# of the epochs')
    parser.add_argument('--schedule', type=int, nargs='+', default=[400, 800, 1000],
                        help='Decrease learning rate at these epochs.')
    parser.add_argument('--batch_size', nargs='?', type=int, default=16,
                        help='Batch Size')
    parser.add_argument('--lr', nargs='?', type=float, default=7*1e-3,
                        help='Learning Rate')
    parser.add_argument('--show', nargs='?', type=int, default=20,
                        help='show step')
    parser.add_argument('--num_worker', nargs='?', type=int, default=0,
                        help='num_worker to get data')
    parser.add_argument('--resume', default='./checkpoints/DB_deformable_resnet18_bs_16_ep_1200/DB.pth.tar', type=str, metavar='PATH',  #
                        help='model to load')
    parser.add_argument('--checkpoints', default='', type=str, metavar='PATH',  #
                        help='path to save checkpoints (default: checkpoints)')
    args = parser.parse_args()

    train_net(args)