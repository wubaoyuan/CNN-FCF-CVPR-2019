import torch.nn as nn
import math
import torch
from .modules import *
from torch.autograd import Variable
import time


def conv3x3(mode, in_planes, out_planes, k_in_mask, k_out_mask, output, stride=1):
    if mode=='finetune' or  mode=='full':
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
    if mode=='sparse':
        return SparseConv2d(in_planes, out_planes, kernel_size=3, k=out_planes , rho_init=0.001,  rho_maximum=6, mu=1.005, stride=stride,padding=1, bias=False)
    if mode=='inference':
        return PrunedConv2d(in_planes, out_planes, k_in_mask, k_out_mask, output, kernel_size=3, stride=stride, padding=1, bias=False)
        
'''
basic block, which is used for ResNet-18,34
'''
class BasicBlock(nn.Module):
    def __init__(self, mode, inplanes, cfg, ones, outputs, stride=1, downsample=None):
        super(BasicBlock, self).__init__()    
        
        self.mode = mode
        self.ones = ones

        self.conv1 = conv3x3(mode, inplanes, cfg[0], ones[0], ones[1], outputs[0], stride)   #cfg:64  
        self.bn1 = nn.BatchNorm2d(cfg[0], momentum=1)
        self.relu = nn.ReLU(inplace=True) 
        self.conv2 = conv3x3(mode, cfg[0], cfg[1], ones[1], ones[2], outputs[1])
        self.bn2 = nn.BatchNorm2d(cfg[1], momentum=1)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        
        t0 = time.time()
        out = self.conv1(x)
        t1 = time.time()
        self.conv1.conv_time = t1 -t0
        

        out = self.bn1(out)
        out = self.relu(out)
    

        t0 = time.time()
        out = self.conv2(out)
        self.conv2.conv_time = time.time() -t0
            
        out = self.bn2(out)
                
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out


class ResNet_BasicBlock(nn.Module):

    def __init__(self, mode, cfg=None, ones_list=None, output_list=None):
        super(ResNet_BasicBlock, self).__init__()
        
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=1 )
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  #56*56*64
        
        self.layer1 = self._make_layer(mode, BasicBlock, cfg[0:3*2], ones_list[0:3*2+1], output_list[0:3*2], 3)
        self.layer2 = self._make_layer(mode, BasicBlock, cfg[3*2:7*2], ones_list[3*2:7*2+1], output_list[3*2:7*2], 4, stride=2)
        self.layer3 = self._make_layer(mode, BasicBlock, cfg[7*2:13*2], ones_list[7*2:13*2+1], output_list[7*2:13*2], 6, stride=2)
        self.layer4 = self._make_layer(mode, BasicBlock, cfg[13*2:16*2], ones_list[13*2:16*2+1], output_list[13*2:16*2], 3, stride=2)
        
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(cfg[-1], 1000)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, mode, block, filters, ones,  outputs,  n, stride=1):
        downsample = None
        if stride != 1:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, filters[1], kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(filters[1], momentum=1),
            )

        layers = []
        layers.append(block(mode,self.inplanes, filters[0:2], ones[0:2+1], outputs[0:2],  stride, downsample))
        self.inplanes=filters[1]
        
        for i in range(1, n):
            layers.append(block(mode,self.inplanes, filters[i*2:(i+1)*2], ones[i*2:(i+1)*2+1], outputs[i*2:(i+1)*2]))
            self.inplanes = filters[i*2+1]
        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
    
    
'''
bottleneck block, which is used for ResNet-50,101
'''
    
class Bottleneck(nn.Module):
    def __init__(self, mode, inplanes, cfg, ones, outputs, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        
        self.mode = mode
        self.ones = ones
        
        if mode == 'finetune' or mode == 'full':
            self.conv1 = nn.Conv2d(inplanes, cfg[0], kernel_size=1, bias=False)
            self.conv2 = nn.Conv2d(cfg[0], cfg[1], kernel_size=3, stride=stride, padding=1, bias=False)
            self.conv3 = nn.Conv2d(cfg[1], cfg[2], kernel_size=1, bias=False)

        if mode == 'sparse':
            self.conv1 = SparseConv2d(inplanes, cfg[0], kernel_size=1, k=cfg[0], rho_init=0.001,  rho_maximum=6, mu=1.005,bias=False)
            self.conv2 = SparseConv2d(cfg[0], cfg[1], kernel_size=3, k=cfg[1], rho_init=0.001,  rho_maximum=6, mu=1.005, stride=stride,padding=1, bias=False)
            self.conv3 = SparseConv2d(cfg[1], cfg[2], kernel_size=1, k=cfg[2], rho_init=0.001,  rho_maximum=6, mu=1.005, bias=False)
        
        if mode == 'inference':
            self.conv1 = PrunedConv2d(inplanes, cfg[0], ones[0], ones[1],  outputs[0], kernel_size=1, bias=False)
            self.conv2 = PrunedConv2d(cfg[0], cfg[1], ones[1], ones[2],  outputs[1], kernel_size=3, stride=stride, padding=1, bias=False)
            self.conv3 = PrunedConv2d(cfg[1], cfg[2], ones[2], ones[3],  outputs[2], kernel_size=1, bias=False)
        
        self.bn1 = nn.BatchNorm2d(cfg[0])
        self.bn2 = nn.BatchNorm2d(cfg[1])
        self.bn3 = nn.BatchNorm2d(cfg[2])
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        torch.cuda.synchronize()
        t0 = time.time()
        out = self.conv1(x)
        torch.cuda.synchronize()
        self.conv1.conv_time = time.time() -t0
            
        out = self.bn1(out)
        out = self.relu(out)

        torch.cuda.synchronize()
        t0 = time.time()
        out = self.conv2(out)
        torch.cuda.synchronize()
        self.conv2.conv_time = time.time() -t0
            
        out = self.bn2(out)
        out = self.relu(out)
        
        torch.cuda.synchronize()
        t0 = time.time()
        out = self.conv3(out)
        torch.cuda.synchronize()
        self.conv3.conv_time = time.time() -t0
        
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet_Bottleneck(nn.Module):

    def __init__(self, mode, cfg=None, ones_list=None, output_list=None):
        self.inplanes = 64
        super(ResNet_Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(mode, Bottleneck, cfg[0:3*3], ones_list[0:3*3+1], output_list[0:3*3], 3)
        self.layer2 = self._make_layer(mode, Bottleneck, cfg[3*3:7*3], ones_list[3*3:7*3+1], output_list[3*3:7*3], 4, stride=2)
        self.layer3 = self._make_layer(mode, Bottleneck, cfg[7*3:13*3], ones_list[7*3:13*3+1], output_list[7*3:13*3], 6, stride=2)
        self.layer4 = self._make_layer(mode, Bottleneck, cfg[13*3:16*3], ones_list[13*3:16*3+1], output_list[13*3:16*3],3, stride=2)

        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(cfg[-1], 1000)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, mode, block, filters, ones,  outputs, n, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != filters[2]:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, filters[2], kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(filters[2]),
            )
        layers = []
        layers.append(block( mode, self.inplanes, filters[0:3], ones[0:3+1], outputs[0:3], stride, downsample))
        self.inplanes = filters[2]
        for i in range(1, n):
            layers.append(block(mode, self.inplanes, filters[i*3:(i+1)*3], ones[i*3:(i+1)*3+1], outputs[i*3:(i+1)*3]))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        

        x = self.layer1(x)      
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x



'''
get models
'''
def resnet50(mode, cfg=None, ones_list=None, output_list=None):
    if cfg is None:
        cfg = [[64, 64, 256]*3, [128, 128, 512]*4, [256, 256, 1024]*6, [512, 512, 2048]*3]
        cfg = [item for sub_list in cfg for item in sub_list]
        
    if ones_list == None:
        ones_list =list(range(len(cfg)+1))
        
    if output_list == None:
        output_list =list(range(len(cfg)+1))    
    
    model = ResNet_Bottleneck(mode, cfg,  ones_list, output_list)
    return model


def resnet34(mode, cfg=None, ones_list=None, output_list=None):
    if cfg is None:
        cfg = [[64, 64]*3, [128, 128]*4, [256, 256]*6, [512, 512]*3]
        cfg = [item for sub_list in cfg for item in sub_list]
    
    if ones_list == None:
        ones_list =list(range(len(cfg)+1))
        
    if output_list == None:
        output_list =list(range(len(cfg)+1))
        
    
    model = ResNet_BasicBlock(mode, cfg, ones_list, output_list)
    return model

