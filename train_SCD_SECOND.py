import os
import time
import random
import numpy as np
import torch.nn as nn
import torch.autograd
from skimage import io
from torch import optim
import torch.nn.functional as F
# from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
working_path = os.path.dirname(os.path.abspath(__file__))

from utils.loss import CrossEntropyLoss2d, weighted_BCE_logits, ChangeSimilarity
from utils.utils import accuracy, SCDD_eval_all, AverageMeter

#Data and model choose
###############################################
from datasets import RS_ST_Second as RS
from models.STSPNet import STSPNet as Net
NET_NAME = 'STSPNet'
DATA_NAME = 'SECOND'
###############################################    
    
#Training options
###############################################
args = {
    'train_batch_size': 8,
    'val_batch_size': 8,
    'lr': 0.1,
    'epochs': 50,
    'gpu': True,
    'lr_decay_power': 1.5,
    'weight_decay': 5e-4,
    'momentum': 0.9,
    'print_freq': 50,
    'predict_step': 5,
    'pred_dir': os.path.join(working_path, 'results', DATA_NAME),
    'chkpt_dir': os.path.join(working_path, 'checkpoints', DATA_NAME),
    'log_dir': os.path.join(working_path, 'logs', DATA_NAME, NET_NAME),
    'load_path': os.path.join(working_path, 'checkpoints', DATA_NAME, 'pretrained.pth')
}
###############################################

if not os.path.exists(args['log_dir']): os.makedirs(args['log_dir'])
if not os.path.exists(args['pred_dir']): os.makedirs(args['pred_dir'])
if not os.path.exists(args['chkpt_dir']): os.makedirs(args['chkpt_dir'])
# writer = SummaryWriter(args['log_dir'])

def main():        
    net = Net(3, num_classes=RS.num_classes).cuda()
    #net.load_state_dict(torch.load(args['load_path']), strict=False)
        
    train_set = RS.Data('train', random_flip=True)
    train_loader = DataLoader(train_set, batch_size=args['train_batch_size'], num_workers=4, shuffle=True)
    val_set = RS.Data('val')
    val_loader = DataLoader(val_set, batch_size=args['val_batch_size'], num_workers=4, shuffle=False)
    
    criterion = CrossEntropyLoss2d(ignore_index=0).cuda()
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args['lr'], weight_decay=args['weight_decay'], momentum=args['momentum'], nesterov=True)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95, last_epoch=-1)

    train(train_loader, net, criterion, optimizer, scheduler, val_loader)
    # writer.close()
    print('Training finished.')

def train(train_loader, net, criterion, optimizer, scheduler, val_loader):
    bestaccT=0
    bestscoreV=0.0
    bestloss=1.0
    begin_time = time.time()
    all_iters = float(len(train_loader)*args['epochs'])

    curr_epoch=0
    while True:
        torch.cuda.empty_cache()
        net.train()
        #freeze_model(net.FCN)
        start = time.time()
        acc_meter = AverageMeter()
        train_seg_loss = AverageMeter()
        train_bn_loss = AverageMeter()
        train_uc_loss = AverageMeter()
        criterionuc = nn.MSELoss()
        curr_iter = curr_epoch*len(train_loader)
        for i, data in enumerate(train_loader):
            running_iter = curr_iter+i+1
            adjust_lr(optimizer, running_iter, all_iters)
            imgs_A, imgs_B, labels_A, labels_B = data
            if args['gpu']:
                imgs_A = imgs_A.cuda().float()
                imgs_B = imgs_B.cuda().float()
                labels_bn = (labels_A>0).unsqueeze(1).cuda().float()
                labels_A = labels_A.cuda().long()
                labels_B = labels_B.cuda().long()

            optimizer.zero_grad()
            out_change, outputs_A, outputs_B, dist = net(imgs_A, imgs_B)
            
            assert outputs_A.size()[1] == RS.num_classes

            loss_seg1 = criterion(outputs_A, labels_A) * 0.5
            loss_seg4 = criterion(outputs_B, labels_B) * 0.5
            loss_seg = (loss_seg1  + loss_seg4)*0.7  
            zeo = torch.zeros(dist.shape[0],dist.shape[1],dist.shape[2],dist.shape[3]).cuda().float()
            label_unchange = ~labels_bn.bool()
            dist_unchange = dist * label_unchange
            loss_uc = criterionuc(dist_unchange,zeo)*0.02
            loss_bn = weighted_BCE_logits(out_change, labels_bn)
            loss = loss_seg + loss_bn + loss_uc
            loss.backward()
            optimizer.step()

            labels_A = labels_A.cpu().detach().numpy()
            labels_B = labels_B.cpu().detach().numpy()
            outputs_A = outputs_A.cpu().detach()
            outputs_B = outputs_B.cpu().detach()
            change_mask = F.sigmoid(out_change).cpu().detach()>0.5
            preds_A = torch.argmax(outputs_A, dim=1)
            preds_B = torch.argmax(outputs_B, dim=1)
            preds_A = (preds_A*change_mask.squeeze().long()).numpy()
            preds_B = (preds_B*change_mask.squeeze().long()).numpy()
            # batch_valid_sum = 0
            acc_curr_meter = AverageMeter()
            for (pred_A, pred_B, label_A, label_B) in zip(preds_A, preds_B, labels_A, labels_B):
                acc_A, valid_sum_A = accuracy(pred_A, label_A)
                acc_B, valid_sum_B = accuracy(pred_B, label_B)
                acc = (acc_A + acc_B)*0.5
                acc_curr_meter.update(acc)
            acc_meter.update(acc_curr_meter.avg)
            train_seg_loss.update(loss_seg.cpu().detach().numpy())
            train_bn_loss.update(loss_bn.cpu().detach().numpy())
            train_uc_loss.update(loss_uc.cpu().detach().numpy())

            curr_time = time.time() - start

            if (i + 1) % args['print_freq'] == 0:
                print('[epoch %d] [iter %d / %d %.1fs] [lr %f] [train seg_loss %.4f bn_loss %.4f acc %.2f]' % (
                    curr_epoch, i + 1, len(train_loader), curr_time, optimizer.param_groups[0]['lr'],
                    train_seg_loss.val, train_bn_loss.val, acc_meter.val*100)) #sc_loss %.4f, train_sc_loss.val, 
                    
        score_v, mIoU_v, Sek_v, acc_v, loss_v = validate(val_loader, net, criterion, curr_epoch)
        if acc_meter.avg>bestaccT: bestaccT=acc_meter.avg
        if score_v>bestscoreV:
            bestscoreV=score_v
            bestaccV=acc_v
            bestloss=loss_v
            torch.save(net.state_dict(), os.path.join(args['chkpt_dir'], NET_NAME+'_%de_mIoU%.2f_Sek%.2f_score%.2f_OA%.2f.pth'\
                %(curr_epoch, mIoU_v*100, Sek_v*100, score_v*100, acc_v*100)) )
        print('Total time: %.1fs Best rec: Train acc %.2f, Val score %.2f acc %.2f loss %.4f' %(time.time()-begin_time, bestaccT*100, bestscoreV*100, bestaccV*100, bestloss))
        curr_epoch += 1
        #scheduler.step()
        if curr_epoch >= args['epochs']:
            return

def validate(val_loader, net, criterion, curr_epoch):
    # the following code is written assuming that batch size is 1
    net.eval()
    torch.cuda.empty_cache()
    start = time.time()

    val_loss = AverageMeter()
    acc_meter = AverageMeter()

    preds_all = []
    labels_all = []
    for vi, data in enumerate(val_loader):
        imgs_A, imgs_B, labels_A, labels_B = data
        if args['gpu']:
            imgs_A = imgs_A.cuda().float()
            imgs_B = imgs_B.cuda().float()
            labels_A = labels_A.cuda().long()
            labels_B = labels_B.cuda().long()

        with torch.no_grad():
            out_change, outputs_A, outputs_B, dist = net(imgs_A, imgs_B)
            loss_A = criterion(outputs_A, labels_A)
            loss_B = criterion(outputs_B, labels_B)
            loss = loss_A * 0.5 + loss_B * 0.5
        val_loss.update(loss.cpu().detach().numpy())

        labels_A = labels_A.cpu().detach().numpy()
        labels_B = labels_B.cpu().detach().numpy()
        outputs_A = outputs_A.cpu().detach()
        outputs_B = outputs_B.cpu().detach()
        change_mask = F.sigmoid(out_change).cpu().detach()>0.5
        preds_A = torch.argmax(outputs_A, dim=1)
        preds_B = torch.argmax(outputs_B, dim=1)
        preds_A = (preds_A*change_mask.squeeze().long()).numpy()
        preds_B = (preds_B*change_mask.squeeze().long()).numpy()
        for (pred_A, pred_B, label_A, label_B) in zip(preds_A, preds_B, labels_A, labels_B):
            acc_A, valid_sum_A = accuracy(pred_A, label_A)
            acc_B, valid_sum_B = accuracy(pred_B, label_B)
            preds_all.append(pred_A)
            preds_all.append(pred_B)
            labels_all.append(label_A)
            labels_all.append(label_B)
            acc = (acc_A + acc_B)*0.5
            acc_meter.update(acc)

    
    score, IoU_mean, Sek = SCDD_eval_all(preds_all, labels_all, RS.num_classes)

    curr_time = time.time() - start
    print('%.1fs Val loss: %.2f Score: %.2f IoU: %.2f Sek: %.2f Accuracy: %.2f'\
    %(curr_time, val_loss.average(), score*100, IoU_mean*100, Sek*100, acc_meter.average()*100))


    return score, IoU_mean, Sek, acc_meter.avg, val_loss.avg

def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()            

def adjust_lr(optimizer, curr_iter, all_iter, init_lr=args['lr']):
    scale_running_lr = ((1. - float(curr_iter) / all_iter) ** args['lr_decay_power'])
    running_lr = init_lr * scale_running_lr

    for param_group in optimizer.param_groups:
        param_group['lr'] = running_lr
        
if __name__ == '__main__':
    main()