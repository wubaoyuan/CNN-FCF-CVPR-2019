import torch.nn as nn
import torch.nn.functional as F
from .modules import *
from torch.autograd import Variable
import torch


class ResidualBlock(nn.Module):
    def __init__(self, mode, inchannel, cfg, ones, outputs, stride=1):
        super(ResidualBlock, self).__init__()
        
        if mode=='finetune' or mode=='full':
            self.left = nn.Sequential( 
                nn.Conv2d(inchannel, cfg[0], kernel_size=3, stride=stride, padding=1,bias=False),
                nn.BatchNorm2d(cfg[0]),
                nn.ReLU(inplace=True),
                nn.Conv2d(cfg[0], cfg[1], kernel_size=3, stride=1, padding=1,bias=False),
                nn.BatchNorm2d(cfg[1])
        )
        
        if mode=='sparse':
            self.left = nn.Sequential( 
                SparseConv2d(inchannel, cfg[0], kernel_size=3, k=cfg[0] , rho_init=0.01,  rho_maximum=6, mu=1.01, stride=stride, padding=1,bias=False),
                nn.BatchNorm2d(cfg[0]),
                nn.ReLU(inplace=True),
                SparseConv2d(cfg[0], cfg[1], kernel_size=3, k=cfg[1] , rho_init=0.01,  rho_maximum=6, mu=1.01, stride=1, padding=1,bias=False),
                nn.BatchNorm2d(cfg[1])
        )
            
        if mode=='inference':
            self.left = nn.Sequential( 
                PrunedConv2d(inchannel, cfg[0], ones[0], ones[1], outputs[0], kernel_size=3, stride=stride, padding=1,bias=False),
                nn.BatchNorm2d(cfg[0]),
                nn.ReLU(inplace=True),
                PrunedConv2d(cfg[0], cfg[1], ones[1], ones[2], outputs[1], kernel_size=3, stride=1, padding=1,bias=False),
                nn.BatchNorm2d(cfg[1])
        )
            
        
        self.shortcut = None
        if stride != 1 or inchannel !=cfg[1]:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, cfg[1], kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(cfg[1])
            )

    def forward(self, x):
        residual = x
        out = self.left(x)
        if self.shortcut is not None:
            residual = self.shortcut(x)
            
        out += residual
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, mode, cfg, block_num, ones_list, output_list):
        super(ResNet, self).__init__()
            
        self.inchannel = 16        
        self.conv1 = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(16),
                nn.ReLU(),
            )   #32*32*16
        
        self.layer1 = self.make_layer(mode, ResidualBlock, cfg[0:block_num*2], block_num, ones_list[0:block_num*2+1], output_list[0:block_num*2], stride=1)
        self.layer2 = self.make_layer(mode, ResidualBlock, cfg[block_num*2:block_num*2*2], block_num, ones_list[block_num*2:block_num*2*2+1], output_list[block_num*2:block_num*2*2], stride=2)
        self.layer3 = self.make_layer(mode, ResidualBlock, cfg[block_num*2*2:block_num*3*2], block_num, ones_list[block_num*2*2:block_num*3*2+1], output_list[block_num*2*2:block_num*3*2], stride=2)
        
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = nn.Linear(cfg[-1], 10)

    def make_layer(self, mode, block, filters, n, ones, outputs, stride):
        layers = []
        layers.append(block(mode,self.inchannel, filters[0:2], ones[0:2+1], outputs[0:2], stride))
        self.inchannel=filters[1]

        for i in range(1,n):
            layers.append(block(mode, self.inchannel, filters[i*2:(i+1)*2], ones[i*2:(i+1)*2+1], outputs[i*2:(i+1)*2]))
            self.inchannel = filters[i*2+1]
            
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)   #32*32*16
        out = self.layer1(out)  #32*32*16
        out = self.layer2(out)  #16*16*32
        out = self.layer3(out)  #8*8*64
        
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

def resnet20(mode, cfg=None, ones_list=None, output_list=None):
    if cfg is None:
        cfg = [[16, 16]*3, [32, 32]*3, [64, 64]*3]
        cfg = [item for sub_list in cfg for item in sub_list]
    
    if ones_list == None:
        ones_list =list(range(len(cfg)+1))
        
    if output_list == None:
        output_list =list(range(len(cfg)+1))     
    
    model = ResNet(mode, cfg, 3, ones_list, output_list)
    return model

def resnet32(mode, cfg=None, ones_list=None, output_list=None):
    if cfg is None:
        cfg = [[16, 16]*5, [32, 32]*5, [64, 64]*5]
        cfg = [item for sub_list in cfg for item in sub_list]
    
    if ones_list == None:
        ones_list =list(range(len(cfg)+1))
        
    if output_list == None:
        output_list =list(range(len(cfg)+1))     
    
    model = ResNet(mode, cfg, 5, ones_list, output_list)
    return model

def resnet56(mode, cfg=None, ones_list=None, output_list=None):
    if cfg is None:
        cfg = [[16, 16]*9, [32, 32]*9, [64, 64]*9]
        cfg = [item for sub_list in cfg for item in sub_list]
    
    if ones_list == None:
        ones_list =list(range(len(cfg)+1))
        
    if output_list == None:
        output_list =list(range(len(cfg)+1))     
    
    model = ResNet(mode, cfg, 9, ones_list, output_list)
    return model	

def resnet110(mode, cfg=None, ones_list=None, output_list=None):
    if cfg is None:
        cfg = [[16, 16]*18, [32, 32]*18, [64, 64]*18]
        cfg = [item for sub_list in cfg for item in sub_list]
    
    if ones_list == None:
        ones_list =list(range(len(cfg)+1))
        
    if output_list == None:
        output_list =list(range(len(cfg)+1))       
    
    model = ResNet(mode, cfg, 18 , ones_list, output_list)
    return model	