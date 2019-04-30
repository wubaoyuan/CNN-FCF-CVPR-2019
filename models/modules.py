import math
import numpy as np
import scipy.sparse as sparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from torch.autograd import Variable        
import time



        
'''
sparseconv2d
'''

class SparseConv2d(nn.Module):

    def __init__(self, in_channels, out_channels,  kernel_size, k, rho_init, rho_maximum, mu,  stride=1, padding=0, bias=True):
        super(SparseConv2d, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size =  _pair(kernel_size)
        self.stride =  _pair(stride)
        self.padding = _pair(padding)
        

        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, *self.kernel_size))
        self.v = nn.Parameter(torch.Tensor(out_channels))
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
        
        #admm paras
        self.k=k
        self.rho_init=rho_init 
        self.rho=rho_init          
        self.rho_maximum=rho_maximum
        self.mu=mu
        
        self.y1=np.zeros([out_channels,1])
        self.y2=np.zeros([out_channels,1])
        self.z1=np.zeros([out_channels,1])
        self.z2=np.zeros([out_channels,1])
        self.v_np=np.zeros([out_channels,1])
               
        self.P=sparse.csc_matrix(np.eye(self.out_channels))
        self.q=np.zeros([self.out_channels,1])
        self.E=sparse.csc_matrix(np.vstack([np.eye(self.out_channels), np.ones([self.out_channels,1]).transpose()]))
        self.l=np.vstack([np.zeros([self.out_channels,1]), self.k*np.ones([1,1])])
        self.u=np.vstack([np.ones([self.out_channels,1]),  self.k*np.ones([1,1])])
        

        
    def reset_parameters(self):
        #init weight
        stdv = math.sqrt(2. / (sum(self.weight.size())))
        self.weight.data.normal_(0, stdv)
        
        #init bias
        if self.bias is not None:
            self.bias.data.zero_()
        
        #init v
        v_shape=self.v.data.numpy().shape
        np_v=np.ones(v_shape)
        self.v.data=torch.from_numpy(np_v).float()

    def forward(self, input):           
        return F.conv2d(input, torch.diag(self.v).mm(self.weight.view(self.out_channels,self.in_channels*self.kernel_size[0]*self.kernel_size[1])).view_as(self.weight), self.bias, self.stride, self.padding)



'''
prunedconv2d
'''
class PrunedConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, k_in_mask, k_out_mask, output_width, kernel_size, stride=1, padding=0, bias=True):
        super(PrunedConv2d, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size =  _pair(kernel_size)
        self.stride =  _pair(stride)
        self.padding = _pair(padding)
        

        self.k_in = len(k_in_mask)
        self.k_out = len(k_out_mask)
        self.k_in_mask = k_in_mask
        self.k_out_mask = k_out_mask
        self.output_width = output_width

        self.weight = nn.Parameter(torch.Tensor(self.k_out, self.k_in, *self.kernel_size))   
        self.output = Variable(torch.zeros(50, self.out_channels, self.output_width, self.output_width))#.cuda() 
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.k_out))
        else:
            self.register_parameter('bias', None)
        

    def forward(self, input):
        #squeeze
        t0 = time.time()
        temp_in = torch.index_select(input, 1, self.k_in_mask)
        self.squeeze_time = time.time() -t0
        
        #conv
        temp_out = F.conv2d(temp_in, self.weight, self.bias, self.stride, self.padding)
        
        #expand
        t0 = time.time()
        out = self.output[0:temp_out.shape[0]].index_copy(1, self.k_out_mask, temp_out)
        self.expand_time = time.time() -t0
        
        
        return out
    
        #return self.output.index_copy(1, self.k_out_mask, F.conv2d(torch.index_select(input, 1, self.k_in_mask), self.weight, self.bias, self.stride, self.padding))