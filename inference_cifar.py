import os
os.environ["CUDA_VISIBLE_DEVICES"]= '7'

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from datetime import datetime
import numpy as np


from tools import *
from models import *
import time
import argparse


parser = argparse.ArgumentParser(description='CIFAR-10 inference')

parser.add_argument('--batch-size', default=50, type=int, metavar='N',
                    help='mini-batch size')
parser.add_argument('--finetune-model', default='./checkpoints/inference/resnet20_finetune_best_025.pth.tar', type=str,
                    help='finetune model checkpoint')
parser.add_argument('--model', default='resnet20', type=str,
                    help='choose the training mode')
parser.add_argument('--n', default='6', type=int,
                    help='scale factor for different resnet model(6,10,18,36)')



def main():
    args = parser.parse_args()
    training_models = {'resnet20':resnet20, 'resnet32':resnet32, 'resnet56':resnet56, 'resnet110':resnet110}
    

    #finetune model with zeros
    finetune_model = training_models[args.model](mode='finetune')
    checkpoint = torch.load(args.finetune_model)  
    finetune_model.load_state_dict(checkpoint['state_dict'])
    finetune_model = finetune_model.eval().cuda() 

    ones_list = [Variable(torch.from_numpy(np.array(list(range(16))))).cuda()]
    for i, m in enumerate(finetune_model.modules()):
        if isinstance(m, nn.Conv2d) and m.kernel_size == (3,3) and m.in_channels!=3:
            out_channels = m.weight.data.shape[0]
            output_norms = torch.norm(m.weight.data.view(out_channels,-1), p=1, dim=1)
            output_ones = torch.gt(output_norms,0).nonzero()
            ones_list.append(Variable(output_ones.squeeze()).cuda())

            
    
    #squeeze-conv-expand model for inference
    cfg_list =[]
    cfgs =[[32]*args.n, [16]*args.n, [8]*args.n]  #6,10,18,36
    for cfg in cfgs :
        cfg_list.extend(cfg)
        
    pruned_model=training_models[args.model](mode='inference', ones_list=ones_list, output_list = cfg_list)
    pruned_model = pruned_model.eval().cuda()

    #assign parameters
    i=1
    for [m0, m1] in zip(finetune_model.modules(), pruned_model.modules()):   #assign value for pruned_model
        if isinstance(m0, nn.Conv2d):
            if m0.kernel_size == (1,1):  #down-sampling convolution.
                m1.weight.data = m0.weight.data.clone()

            else:
                if isinstance(m1, PrunedConv2d): 
                    w = torch.index_select(m0.weight.data, 0, ones_list[i].data)
                    m1.weight.data = torch.index_select(w, 1, ones_list[i-1].data)
                    i = i+1
                    
                else:  #first conv layer
                    m1.weight.data = m0.weight.data.clone()
                
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
    pruned_parameters = sum([param.nelement() for param in pruned_model.parameters()])
    full_parameters = sum([param.nelement() for param in full_model.parameters()])
    print('parameters sparse rate:{}'.format(1-float(pruned_parameters)/full_parameters))

    #theoretical reduced flops rate
    test_data = Variable(torch.randn(1,3,32,32)).cuda()
    full_flops = calculate_flops(full_model, test_data)
    pruned_flops = calculate_flops(pruned_model, test_data)
    print('flops sparse rate:{}'.format(1-float(pruned_flops) / full_flops))
    
    
    testloader = cifar10_testdata(args.batch_size)
    criterion = nn.CrossEntropyLoss()
    validate(testloader, finetune_model, criterion)
    validate(testloader, pruned_model, criterion)


def validate(testloader, model, criterion):
    disc_loss = 0.
    disc_acc = 0.
    testset =0
    for data in testloader:
        img, label = data
        img = Variable(img).cuda()
        label = Variable(label).cuda()

        out = model(img)
        loss = criterion(out, label)

        disc_loss += loss.data[0] * label.size(0)
        _, pred = torch.max(out, 1)
        num_correct = (pred == label).sum()
        disc_acc += num_correct.data[0]
        testset += label.size(0)
        
    print('Test Loss : {:.6f},Test Acc: {:.6f}'.format(disc_loss /testset, disc_acc /testset))
    return disc_acc/testset, disc_loss/testset

if __name__ == '__main__':
    main()