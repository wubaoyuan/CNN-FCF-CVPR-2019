import os
os.environ["CUDA_VISIBLE_DEVICES"]= '7'

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from datetime import datetime
import argparse

from functions import *
from models import *

parser = argparse.ArgumentParser(description='Training a cnn-fcf model on CIFAR-10')

parser.add_argument('--epochs', default=30, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--decay-epochs', default=10, type=int,
                    help='number of epochs to decay the learning rate')
parser.add_argument('--print-freq', '-p', default=100, type=int,
                    metavar='N', help='print frequency')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--alpha', default=0.99, type=float,
                    help='the trade-off between the cnn and admm')
parser.add_argument('--sparse-rate', default=0.25, type=float,
                    help='the filter pruning ratio')
parser.add_argument('--model', default='resnet20', type=str,
                    help='choose the model')
parser.add_argument('--training-mode', default='sparse', type=str,
                    help='choose the training mode')
parser.add_argument('--sparse-mode', default='identical_ratio', type=str,
                    help='choose the mode to set sparse rate')
parser.add_argument('--pretrained-model', default='./checkpoints/pretrain/resnet20_cifar_full.pkl', type=str,
                    help='the path to save the best result')
parser.add_argument('--checkpoint-name', default='./checkpoints/fcf/resnet20_sparse_025', type=str,
                    help='the path to save the checkpoint')



best_prec1 = 0
training_models = {'resnet20':resnet20, 'resnet32':resnet32, 'resnet56':resnet56, 'resnet110':resnet110}


def main():
    global args, best_prec1
    args = parser.parse_args()
       
    trainloader = cifar10_traindata(args.batch_size)
    testloader = cifar10_testdata(args.batch_size)
    
    model = training_models[args.model](mode = args.training_mode)
    pretrained_dict = torch.load(args.pretrained_model)
    model_dict = model.state_dict()
    temp_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(temp_dict) 
    model.load_state_dict(model_dict)
    model = model.cuda()
    
    criterion = nn.CrossEntropyLoss()  
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    sparse_k(model,args)
    
    
    train_loss=[]
    train_accuracy=[]
    test_accuracy=[]
    test_loss=[]
    for epoch in range(args.epochs):
        if epoch ==10:   
            for param_group in optimizer.param_groups:
                param_group['lr']=param_group['lr']*0.1
    
        if epoch ==20:   
            for param_group in optimizer.param_groups:
                param_group['lr']=param_group['lr']*0.1
    
        
        prec1_tr,loss_tr = train(args, trainloader, model, criterion, optimizer, epoch)
        train_accuracy.append(prec1_tr)
        train_loss.append(loss_tr)
        
        prec1,loss = validate(testloader, model, criterion)
        test_accuracy.append(prec1)
        test_loss.append(loss)
        
        np_v_list = store_v(model)
        pruning_prec1, _= validate(testloader, model, criterion)
        reload_v(model, np_v_list) 
        
        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        print('best_prec@1:{}'.format(best_prec1))
        
        save_checkpoint(args, {
                        'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'best_prec1': best_prec1,
                        'optimizer' : optimizer.state_dict(),
                        'train_loss': train_loss,
                        'train_accuracy': train_accuracy,
                        'test_accuracy': test_accuracy,
                        'test_loss': test_loss
                    }, is_best)



def train(args, trainloader, model, criterion, optimizer, epoch):
    print('epoch {}'.format(epoch + 1))
    print('*' * 10)
    print(datetime.now())

    model.train()

    running_loss = 0.0
    running_acc = 0.0
    trainset =0
    for i, data in enumerate(trainloader, 1):
        length = len(trainloader)
        img, label = data     #tensor
        img = Variable(img).cuda()
        label = Variable(label).cuda()

        # forward
        out = model(img)
        loss = criterion(out, label)

        #calculate loss and accuracy
        running_loss += loss.data[0] * label.size(0)   # loss*batch_size , the loss of batch i+loss before
        _, pred = torch.max(out, 1)
        num_correct = (pred == label).sum()
        running_acc += num_correct.data[0]
        trainset +=label.size(0)

        # backward
        optimizer.zero_grad()
        loss.backward()

        #update z1,z2,v_admm
        for m in model.modules():
            if isinstance(m, SparseConv2d):
                admm_update1(m, args.alpha)

        #update gradient
        optimizer.step()

        #update y1,y2,rho
        for m in model.modules():
            if isinstance(m, SparseConv2d):
                admm_update2(m,True)

        if i % args.print_freq == 0:
            print('[epoch:{}/{} , iter:{}/{}] Loss: {:.6f}, Acc: {:.6f}'.format(epoch + 1, args.epochs, i, length, running_loss / (args.batch_size * i),running_acc / (args.batch_size * i)))

    print('Finish {} epoch, Loss: {:.6f}, Acc: {:.6f}'.format(epoch + 1, running_loss / trainset, running_acc / trainset))
    return running_acc / trainset, running_loss / trainset



def validate(testloader, model, criterion):
    model.eval()

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



def save_checkpoint(args, state, is_best):
    torch.save(state, args.checkpoint_name+'.pth.tar')



if __name__ == '__main__':
    main()