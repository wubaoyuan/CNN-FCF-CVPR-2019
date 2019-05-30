import os
os.environ["CUDA_VISIBLE_DEVICES"] = '7'

import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np

from functions import *
from models import *

import argparse


parser = argparse.ArgumentParser(description='Inference a cnn model on ImageNet')

parser.add_argument('-b', '--batch-size', default=50, type=int,
                    metavar='N', help='mini-batch size')
parser.add_argument('--finetune-model', default='./checkpoints/inference/finetune_resnet34_best_same025.pth.tar', type=str,
                    help='finetune model checkpoint')
parser.add_argument('--model', default='resnet34', type=str,
                    help='choose the training mode')


def main():
    training_models = {'resnet34':resnet34, 'resnet50':resnet50}
    args = parser.parse_args()

    #finetune model with zeros
    finetune_model = training_models[args.model](mode='finetune')
    checkpoint =torch.load(args.finetune_model)  
    new_dict={}
    for k, v in checkpoint['state_dict'].items():
        new_key=k[7:]
        new_dict[new_key]=checkpoint['state_dict'][k]
    finetune_model.load_state_dict(new_dict)
    finetune_model = finetune_model.eval().cuda() 


    ones_list = [Variable(torch.from_numpy(np.array(list(range(64))))).cuda()]
    for i, m in enumerate(finetune_model.modules()):
        if isinstance(m, nn.Conv2d) and m.kernel_size == (3,3):
            out_channels = m.weight.data.shape[0]
            norms = torch.norm(m.weight.data.view(out_channels,-1), p=1, dim=1)
            ones = torch.gt(norms,0).nonzero()
            ones_list.append(Variable(ones.squeeze()).cuda())


    #squeeze-conv-expand model
    cfg_list =[]
    cfgs =[[56]*6, [28]*8, [14]*12, [7]*6]
    for cfg in cfgs :
        cfg_list.extend(cfg)
        
    inference_model=training_models[args.model](mode='inference', ones_list=ones_list, output_list = cfg_list)
    inference_model = inference_model.eval().cuda()
    
    #assign parameters
    i=1
    for [m0, m1] in zip(finetune_model.modules(), inference_model.modules()):   #assign value for inference_model
        if isinstance(m0, nn.Conv2d):
            if m0.kernel_size == (1,1):  #down-sampling convolution.
                m1.weight.data = m0.weight.data.clone()
                
            elif m0.kernel_size == (7,7):
                m1.weight.data = m0.weight.data.clone()
                
            else:  #3*3 conv2d
                w = torch.index_select(m0.weight.data, 0, ones_list[i].data)
                m1.weight.data = torch.index_select(w, 1, ones_list[i-1].data)
                i = i+1

                
        elif isinstance(m0, nn.BatchNorm2d):
            m1.weight.data = m0.weight.data.clone()
            m1.bias.data = m0.bias.data.clone()
            m1.running_mean = m0.running_mean.clone()
            m1.running_var = m0.running_var.clone()
            
        elif isinstance(m0, nn.Linear):
            m1.weight.data = m0.weight.data.clone()
            m1.bias.data = m0.bias.data.clone()
    
    
    
    #theoretical compressing rate
    full_model = training_models[args.model](mode='full').cuda()
    full_parameters = sum([param.nelement() for param in full_model.parameters()])
    inference_parameters = sum([param.nelement() for param in inference_model.parameters()])
    print('sparse rate:{}'.format(1-float(inference_parameters)/full_parameters))

    #theoretical reduced flops rate
    test_data = Variable(torch.randn(1,3,224,224)).cuda()
    full_flops = calculate_flops(full_model, test_data)
    inference_flops = calculate_flops(inference_model, test_data)
    print('flops sparse rate:{}'.format(1-float(inference_flops) / full_flops))


    #validation
    testloader = imagenet_testdata(args.batch_size)
    criterion = nn.CrossEntropyLoss()
    validate(testloader, finetune_model, criterion)
    validate(testloader, inference_model, criterion)



def validate(val_loader, model, criterion):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    for i, (input, target) in enumerate(val_loader):
        input_var = torch.autograd.Variable(input, volatile=True).cuda()
        target_var = torch.autograd.Variable(target, volatile=True).cuda()

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target_var.data, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))
        
    print('Test: *Loss {loss.avg:.4f} \tPrec@1 {top1.avg:.3f}'.format(loss=losses, top1=top1))
    return top1.avg,losses.avg



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


if __name__=='__main__':
    main()