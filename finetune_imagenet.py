import argparse
from datetime import datetime
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'

import torch
import torch.nn as nn


from functions import *
from models import *


parser = argparse.ArgumentParser(description='Finetune a cnn model on ImageNet')

parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--decay-epochs', default=30, type=int,
                    help='number of epochs to decay the learning rate')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay')
parser.add_argument('--print-freq', '-p', default=100, type=int,
                    metavar='N', help='print frequency')
parser.add_argument('--model', default='resnet34', type=str,
                    help='choose the training mode')

parser.add_argument('--fcf-checkpoint', default='./checkpoints/fcf/sparse_resnet34_same025.pth.tar', type=str,
                    help='the path to save the best result')
parser.add_argument('--best-checkpoint', default='./checkpoints/inference/finetune_resnet34_best_same025', type=str,
                    help='the path to save the best result')



best_prec1 = 0
training_models = {'resnet50':resnet50, 'resnet34':resnet34}

def main():
    global args, best_prec1
    args = parser.parse_args()
    
    trainloader = imagenet_traindata(args.batch_size)
    testloader = imagenet_testdata(args.batch_size)
    
    
    model = training_models[args.model](mode='full')
    fcf_model = nn.DataParallel(training_models[args.model](mode='sparse'))
    parse_fcf_model(args, fcf_model)
    fcf_dict={}
    for k, v in args.fcf_dict.items():
        new_key=k[7:]
        fcf_dict[new_key]=args.fcf_dict[k]
    model_dict = model.state_dict()
    temp_dict = {k: v for k, v in fcf_dict.items() if k in model_dict}
    model.load_state_dict(temp_dict)
    model = nn.DataParallel(model).cuda()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    

    train_loss=[]
    train_accuracy=[]
    test_accuracy=[]
    test_loss=[]
    for epoch in range(args.epochs):
        adjust_learning_rate(optimizer, epoch)
    
        prec1_tr,loss_tr=train(args, trainloader, model, criterion, optimizer, epoch)
        train_accuracy.append(prec1_tr)
        train_loss.append(loss_tr)
        
        prec1,loss = validate(args, testloader, model, criterion)
        test_accuracy.append(prec1)
        test_loss.append(loss)
        
        sparse_test(args, model)
    
        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        print('best_prec@1:{}'.format(best_prec1))
        
        save_checkpoint(args,
                        {'epoch': epoch + 1,
                         'state_dict': model.state_dict(),
                         'best_prec1': best_prec1,
                         'optimizer' : optimizer.state_dict(),
                         'train_loss': train_loss,
                         'train_accuracy': train_accuracy,
                         'test_accuracy': test_accuracy,
                         'test_loss': test_loss}
                        , is_best)


def train(args, train_loader, model, criterion, optimizer, epoch):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    for i, (input, target) in enumerate(train_loader):
        input_var = torch.autograd.Variable(input).cuda()
        target_var = torch.autograd.Variable(target).cuda()

        # compute output
        zero_process(args, model)
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target_var.data, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()

        # update gradient
        optimizer.step()

        if i % args.print_freq == 0:
            for param_group in optimizer.param_groups:
                lr_rate=param_group['lr']
                break
            time=datetime.now()
            str_time = time.strftime("%Y-%m-%d %H:%M:%S")
            print('Time:{0}   LR:{1}\t  Epoch: [{2}/{3}][{4}/{5}]\t''Loss {loss.val:.4f} ({loss.avg:.4f})\t''Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'.
                format(str_time, lr_rate, epoch, args.epochs, i, len(train_loader), loss=losses, top1=top1))
    return top1.avg,losses.avg


def validate(args, val_loader, model, criterion):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    for i, (input, target) in enumerate(val_loader):
        input_var = torch.autograd.Variable(input, volatile=True).cuda()
        target_var = torch.autograd.Variable(target, volatile=True).cuda()

        # compute output
        zero_process(args, model)
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target_var.data, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

    print('Test: *Loss {loss.avg:.4f} \tPrec@1 {top1.avg:.3f}'.format(loss=losses, top1=top1))
    return top1.avg,losses.avg


def save_checkpoint(args, state, is_best):
    if is_best:
        torch.save(state, args.best_checkpoint+'.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // args.decay_epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()