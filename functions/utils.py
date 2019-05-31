import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable

import sys
sys.path.append("..")
from models import modules



#******************************************
#                train
#******************************************
def sparse_k(model, args):
    
    if args.sparse_mode == 'diffrent_ratio':
        total = 0
        for m in model.modules():
            if isinstance(m, modules.SparseConv2d):
                total += m.weight.data.shape[0]
        print('filters num:{}'.format(total))
    
        ws = torch.zeros(total)
        index = 0
        for m in model.modules():
            if isinstance(m, modules.SparseConv2d):
                size = m.weight.data.shape[0]
                ws[index:(index + size)] = m.weight.data.abs().view(size, -1).mean(1)
                index += size
    
        y, i = torch.sort(ws)
        thre_index = int(total * args.sparse_rate)
        thre = y[thre_index]  # ascending
    
        ks = []
        for k, m in enumerate(model.modules()):
            if isinstance(m, modules.SparseConv2d):
                size = m.weight.data.shape[0]
                weight_compare = m.weight.data.abs().view(size, -1).mean(1)
                mask = weight_compare.gt(thre).float()
                gt_thre = torch.sum(mask)
                ks.append(gt_thre / size)
                print('layer index: {:d} \t total filters: {:d} \t remaining filters: {:d}'.format(k, mask.shape[0],int(torch.sum(mask)))) 
    
    
        print('the goal of sparse k for each layer')
        index = 0
        for i, m in enumerate(model.modules()):
            if isinstance(m, modules.SparseConv2d):  # Avoid the current layer filter number to be 0
                if ks[index] < 0.1:
                    ks[index] = 0.1
                elif ks[index] > 0.9:
                    ks[index] = 0.9
    
                m.k = int(m.k * ks[index])
                m.l = np.vstack([np.zeros([m.out_channels, 1]), m.k * np.ones([1, 1])])
                m.u = np.vstack([np.ones([m.out_channels, 1]), m.k * np.ones([1, 1])])
    
                index = index + 1
                print('layer index:{} \t, sparse target:{}'.format(i, m.k))
        
    elif args.sparse_mode == 'identical_ratio':
        for i, m in enumerate(model.modules()):
            if isinstance(m, modules.SparseConv2d):
                m.k = int(m.k * (1-args.sparse_rate))
                m.l = np.vstack([np.zeros([m.out_channels, 1]), m.k * np.ones([1, 1])])
                m.u = np.vstack([np.ones([m.out_channels, 1]), m.k * np.ones([1, 1])])
        
    else: #no sparse
        for i, m in enumerate(model.modules()):
            if isinstance(m, modules.SparseConv2d):
                m.l = np.vstack([np.zeros([m.out_channels, 1]), m.k * np.ones([1, 1])])
                m.u = np.vstack([np.ones([m.out_channels, 1]), m.k * np.ones([1, 1])])
                

def store_v(model):
    np_v_list = []
    for k, m in enumerate(model.modules()):
        if isinstance(m, modules.SparseConv2d):
            np_v = m.v.data.cpu().numpy().reshape(-1)
            np_v_list.append(np_v)

            zeros = np.squeeze(np.argwhere(np_v < 1e-1))
            np_v[zeros] = 0
            zeros = np.squeeze(np.argwhere(np_v > 1e-1))
            np_v[zeros] = 1
            m.v.data = torch.from_numpy(np_v).float().cuda()
    
    return np_v_list


def reload_v(model, np_v_list):
    index = 0
    for i, m in enumerate(model.modules()):
        if isinstance(m, modules.SparseConv2d):
            m.v.data = torch.from_numpy(np_v_list[index]).float().cuda()
            index += 1


#******************************************
#                finetuning
#******************************************
def parse_fcf_model(args, fcf_model):
    checkpoint = torch.load(args.fcf_checkpoint)
    fcf_model.load_state_dict(checkpoint['state_dict'])
    
    args.fcf_index_list = []
    args.fcf_zeros_list = []

    for i, m in enumerate(fcf_model.modules()):
        if isinstance(m, modules.SparseConv2d):
            np_v = m.v.data.cpu().numpy().reshape(-1)
            zeros = np.squeeze(np.argwhere(np_v < 1e-1))
            args.fcf_index_list.append(i)
            args.fcf_zeros_list.append(zeros)
 
    args.fcf_dict = checkpoint['state_dict']


def zero_process(args, model):       
    model_list=list(model.modules())
    
    index=0
    end_mask=args.fcf_zeros_list[index].tolist()
    start_mask=[]  
    
    for i in args.fcf_index_list:
        m1=model_list[i]
        if isinstance(m1, nn.Conv2d):
            m1_tensor = m1.weight.data.cpu().numpy()
            m1_tensor[:, start_mask, :, :]=0
            m1_tensor[end_mask, :, :, :]=0
            m1.weight.data=torch.from_numpy(m1_tensor).float().cuda()       
            
                         
            start_mask=end_mask
            index += 1
            if index < len(args.fcf_zeros_list):
                end_mask=args.fcf_zeros_list[index]


def sparse_test(args, model):
    total=0
    for m in model.modules():
        if isinstance(m,nn.Conv2d):
            np_weight=m.weight.data.cpu().numpy().reshape(-1)
            zeros=np.squeeze(np.argwhere(np_weight==0))
            total+=len(zeros)
                
    model_parameters = sum([param.nelement() for param in model.parameters()])
    print('sparse rate:{}'.format(float(total)/model_parameters))



#******************************************
#                inference
#******************************************
# code from https://github.com/simochen/model-tools/blob/master/pytorch_tools.py
    
def calculate_flops(model, testdata):
    prods = {}
    def save_hook(name):
        def hook_per(self, input, output):
            prods[name] = np.prod(input[0].shape)
        return hook_per

    list_1=[]
    def simple_hook(self, input, output):
        list_1.append(np.prod(input[0].shape))
        
    list_2={}
    def simple_hook2(self, input, output):
        list_2['names'] = np.prod(input[0].shape)
    
    
    list_prunedconv=[]
    def prunedconv_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()
        kernel_ops = self.kernel_size[0] * self.kernel_size[1] * self.k_in
        bias_ops = 1 if self.bias is not None else 0
        params = self.k_out * (kernel_ops + bias_ops)
        flops = batch_size * params * output_height * output_width
        list_prunedconv.append(flops)
    
    
    list_conv=[]
    def conv_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()
        kernel_ops = self.kernel_size[0] * self.kernel_size[1] * self.in_channels
        bias_ops = 1 if self.bias is not None else 0
        params = output_channels * (kernel_ops + bias_ops)
        flops = batch_size * params * output_height * output_width
        list_conv.append(flops)


    list_linear=[] 
    def linear_hook(self, input, output):
        batch_size = input[0].size(0) if input[0].dim() == 2 else 1
        weight_ops = self.weight.nelement()
        bias_ops = self.bias.nelement()
        flops = batch_size * (weight_ops + bias_ops)
        list_linear.append(flops)

    list_bn=[] 
    def bn_hook(self, input, output):
        list_bn.append(input[0].nelement())

    list_relu=[] 
    def relu_hook(self, input, output):
        list_relu.append(input[0].nelement())

    list_pooling=[]
    def pooling_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()
        kernel_ops = self.kernel_size * self.kernel_size
        bias_ops = 0
        params = output_channels * (kernel_ops + bias_ops)
        flops = batch_size * params * output_height * output_width
        list_pooling.append(flops)


    def foo(net):
        for k, m in enumerate(net.modules()):
            if isinstance(m, torch.nn.Conv2d):
                m.register_forward_hook(conv_hook)
            if isinstance(m, modules.PrunedConv2d):
                m.register_forward_hook(prunedconv_hook)
            if isinstance(m, torch.nn.Linear):
                m.register_forward_hook(linear_hook)
            if isinstance(m, torch.nn.BatchNorm2d):
                m.register_forward_hook(bn_hook)
            if isinstance(m, torch.nn.ReLU):
                m.register_forward_hook(relu_hook)
            if isinstance(m, torch.nn.MaxPool2d) or isinstance(m, torch.nn.AvgPool2d):
                m.register_forward_hook(pooling_hook)

    
    foo(model)
    out = model(testdata)

    total_flops = (sum(list_conv) + sum(list_prunedconv) + sum(list_linear) + sum(list_bn) + sum(list_relu) + sum(list_pooling))
    
    return total_flops