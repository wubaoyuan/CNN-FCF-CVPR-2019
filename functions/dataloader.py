import os

import torch
import torchvision
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

'''
cifar10
'''
def cifar10_traindata(batch_size):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  #padding & crop 
        transforms.RandomHorizontalFlip(),  #0.5
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), #R,G,B zero-mean
    ])
        
    trainset = torchvision.datasets.CIFAR10(root='../cifar10', train=True, download=True, transform=transform_train) 
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)   
    
    return trainloader


    
def cifar10_testdata(batch_size):    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    testset = torchvision.datasets.CIFAR10(root='../cifar10', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return testloader
    
'''
imagenet
'''
traindir = os.path.join('../imagenet2012', 'train')
valdir = os.path.join('../imagenet2012', 'val')
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
def imagenet_traindata(batch_size):
    trainloader = torch.utils.data.DataLoader(
        datasets.ImageFolder(traindir, transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=batch_size,
        shuffle=True,
        num_workers=16, 
        pin_memory=True)
    return trainloader

def imagenet_testdata(batch_size):
    testloader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=batch_size,
        shuffle=False,
        num_workers=16, 
        pin_memory=True)
        
    return testloader
