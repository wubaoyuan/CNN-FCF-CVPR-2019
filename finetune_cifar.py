import os
os.environ["CUDA_VISIBLE_DEVICES"]= '7'

import torch
import torch.nn as nn
from torch.autograd import Variable

from datetime import datetime
import argparse

from functions import *
from models import *

parser = argparse.ArgumentParser(description='Finetune a cnn model on CIFAR-10')

parser.add_argument('--epochs', default=150, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay')
parser.add_argument('--print-freq', '-p', default=100, type=int,
                    metavar='N', help='print frequency')
parser.add_argument('--model', default='resnet20', type=str,
                    help='choose the training mode')

parser.add_argument('--fcf-checkpoint', default='./checkpoints/fcf/resnet20_sparse_025.pth.tar', type=str,
                    help='the path to save the best result')
parser.add_argument('--best-checkpoint', default='./checkpoints/inference/resnet20_finetune_best_025', type=str,
                    help='the path to save the best result')


best_prec1 = 0
training_models = {'resnet20':resnet20, 'resnet32':resnet32, 'resnet56':resnet56, 'resnet110':resnet110}

def main():
    global args, best_prec1
    args = parser.parse_args()
    
    trainloader = cifar10_traindata(args.batch_size)
    testloader = cifar10_testdata(args.batch_size)
    
    model = training_models[args.model](mode='full')
    parse_fcf_model(args, training_models[args.model](mode='sparse'))
    model_dict = model.state_dict()
    temp_dict = {k: v for k, v in args.fcf_dict.items() if k in model_dict}
    model.load_state_dict(temp_dict)
    model = model.cuda()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    

    train_loss=[]
    train_accuracy=[]
    test_accuracy=[]
    test_loss=[]
    for epoch in range(args.epochs):
        if epoch ==60:
            for param_group in optimizer.param_groups:
                param_group['lr']=param_group['lr']*0.1
    
        if epoch ==120:
            for param_group in optimizer.param_groups:
                param_group['lr']=param_group['lr']*0.1
    
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



def train(args, trainloader, model, criterion, optimizer, epoch):
    print('*' * 10)
    print(datetime.now())

    model.train()

    running_loss = 0.0
    running_acc = 0.0
    trainset=0
    for i, data in enumerate(trainloader, 1):
        length = len(trainloader)
        img, label = data     #tensor
        img = Variable(img).cuda()
        label = Variable(label).cuda()

        # forward
        zero_process(args, model)
        out = model(img)
        loss = criterion(out, label)

        #calculate loss and accuracy
        running_loss += loss.data[0] * label.size(0)   # loss*batch_size , the loss of batch i+loss before
        trainset +=label.size(0)
        _, pred = torch.max(out, 1)
        num_correct = (pred == label).sum()
        running_acc += num_correct.data[0]

        # backward
        optimizer.zero_grad()
        loss.backward()

        #update gradient
        optimizer.step()

        if i % args.print_freq == 0:
            print('[epoch:{}/{} , iter:{}/{}] Loss: {:.6f}, Acc: {:.6f}'.format(epoch + 1, args.epochs, i, length, running_loss / (args.batch_size * i),running_acc / (args.batch_size * i)))
        
        
    print('Finish {} epoch, Loss: {:.6f}, Acc: {:.6f}'.format(epoch + 1, running_loss / trainset, running_acc / trainset))
    return running_acc / trainset, running_loss / trainset


def validate(args, testloader, model, criterion):
    model.eval()
    disc_loss = 0.
    disc_acc = 0.
    testset=0
    for data in testloader:
        img, label = data
        img = Variable(img).cuda()
        label = Variable(label).cuda()

        zero_process(args, model)
        out = model(img)
        loss = criterion(out, label)
        disc_loss += loss.data[0] * label.size(0)
        testset +=label.size(0)
        _, pred = torch.max(out, 1)
        num_correct = (pred == label).sum()
        disc_acc += num_correct.data[0]
    print('Test Loss : {:.6f},Test Acc: {:.6f}'.format(disc_loss / testset, disc_acc / testset))
    return disc_acc/testset, disc_loss/testset


def save_checkpoint(args, state, is_best):
    if is_best:
        torch.save(state, args.best_checkpoint+'.pth.tar')


if __name__ == '__main__':
    main()

