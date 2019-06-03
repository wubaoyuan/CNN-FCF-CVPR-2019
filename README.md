# [CNN-FCF](http://openaccess.thecvf.com/content_CVPR_2019/html/Li_Compressing_Convolutional_Neural_Networks_via_Factorized_Convolutional_Filters_CVPR_2019_paper.html)
This is a Pytorch implementation of our paper "Compressing Convolutional Neural Networks via Factorized Convolutional Filters" published in CVPR 2019.
<div align="center">
<img src="/framework.png" width = 95%>
</div>
Above is the overview of the workflow of filter pruning on l-th layer, where the dotted green cubes indicate the pruned filters. (Top): Traditional pruning consists of three sequential stages: pre-training, selecting filters according to a ranking criterion, and fine-tuning. (Bottom): Our method conducts the filter learning and filter selection jointly, through training factorized convolutional filters.

## Table of Contents
- [Requirements](#Requirements)
- [Inference checkpoint files](#Inference-checkpoint-files)
- [Training fcf models](#Training-fcf-models)
  - [Training CIFAR-10](#Training-CIFAR-10)
  - [Training ImageNet](#Training-ImageNet)
- [Finetuning](#Finetuning)
  - [Finetuning CIFAR-10](#Finetuning-CIFAR-10)
  - [Finetuning ImageNet](#Finetuning-ImageNet)
- [Inference](#Inference)
  - [Reproduce the CIFAR-10 results in our paper](#Reproduce-the-CIFAR-10-results-in-our-paper)
  - [Reproduce the ImageNet results in our paper](#Reproduce-the-ImageNet-results-in-our-paper)
- [Running time analysis](#Running-time-analysis)
- [Citation](#Citation)

## Requirements
- Anaconda
- Python 3.6
- PyTorch 0.3.1
- TorchVision 0.2.0
- [OSQP](https://osqp.org/docs/get_started/python.html)

## Inference checkpoint files
The inference models files can be found in [google drive](https://drive.google.com/drive/folders/1VGqpOhAGe9YQcyZTGbzitsLuELjQdsXW?usp=sharing), which can be used to reproduce the results of our paper.

## Training fcf models
#### Training CIFAR-10
```
sh ./scripts/train_cifar_fcf_resnet20.sh

sh ./scripts/train_cifar_fcf_resnet32.sh

sh ./scripts/train_cifar_fcf_resnet56.sh

sh ./scripts/train_cifar_fcf_resnet110.sh
```
#### Training ImageNet
```
sh ./scripts/train_imagenet_fcf_resnet34.sh

sh ./scripts/train_imagenet_fcf_resnet50.sh
```

## Finetuning
Due to the numerical reason, there are still small changes after optimization, so we usually use finetuning to recover the model performance.
#### Finetuning CIFAR-10
```
sh ./scripts/finetune_cifar_resnet20.sh

sh ./scripts/finetune_cifar_resnet32.sh

sh ./scripts/finetune_cifar_resnet56.sh

sh ./scripts/finetune_cifar_resnet110.sh
```
#### Finetuning ImageNet
```
sh ./scripts/finetune_imagenet_resnet34.sh

sh ./scripts/finetune_imagenet_resnet50.sh
```

## Inference

#### Reproduce the CIFAR-10 results in our paper
```
sh ./scripts/inference_cifar_resnet20.sh

sh ./scripts/inference_cifar_resnet32.sh

sh ./scripts/inference_cifar_resnet56.sh

sh ./scripts/inference_cifar_resnet110.sh
```

#### Reproduce the ImageNet results in our paper
```
sh ./scripts/inference_imagenet_resnet34.sh

sh ./scripts/inference_imagenet_resnet50.sh
```

## Running time analysis
We now analyze the running time reduction rate of our method. Considering that the convolution operation of each filter on GPU is independently, and dozens of process are conducted in parallel, we can not get the realtime reduction rate on GPU. The following experiments are conducted on CPU with ResNet34. 

#### Single layer
We first present the single layer running time reduction rate, our customized convolution is composed by squeeze, conv, expand, we also give the proportion of these three operations in the customized convolution, respectively.

<table class="tg">
  <tr>
    <th class="tg-uys7" rowspan="2">Theoretical flops &darr;</th>
    <th class="tg-uys7" rowspan="2">Standard running time &darr;</th>
    <th class="tg-uys7" rowspan="2">Customized running time &darr;</th>
    <th class="tg-uys7" colspan="3">Customized convolution</th>
  </tr>
  <tr>
    <td class="tg-uys7">Squeeze</td>
    <td class="tg-uys7">Conv</td>
    <td class="tg-uys7">Expand</td>
  </tr>
  <tr>
    <td class="tg-uys7">26.04%</td>
    <td class="tg-uys7">17.63%</td>
    <td class="tg-uys7">13.42%</td>
    <td class="tg-uys7">2.76%</td>
    <td class="tg-uys7">92.52%</td>
    <td class="tg-uys7">4.72%</td>
  </tr>
  <tr>
    <td class="tg-uys7">43.75%</td>
    <td class="tg-uys7">34.71%</td>
    <td class="tg-uys7">30.64%</td>
    <td class="tg-uys7">2.91%</td>
    <td class="tg-uys7">91.74%</td>
    <td class="tg-uys7">5.35%</td>
  </tr>
  <tr>
    <td class="tg-uys7">57.75%</td>
    <td class="tg-uys7">42.19%</td>
    <td class="tg-uys7">40.88%</td>
    <td class="tg-uys7">3.01%</td>
    <td class="tg-uys7">91.16%</td>
    <td class="tg-uys7">5.82%</td>
  </tr>
  <tr>
    <td class="tg-uys7">75.00%</td>
    <td class="tg-uys7">65.70%</td>
    <td class="tg-uys7">59.20%</td>
    <td class="tg-uys7">2.27%</td>
    <td class="tg-uys7">92.04%</td>
    <td class="tg-uys7">5.69%</td>
  </tr>
</table>

Note:  
1. Theoretical flops &darr; is denoted as the theoretical flops reduction rate.  
2. Standard running time &darr; is denoted as the standard convolution running time reduction rate.  
3. Customized running time &darr; is denoted as customized convolution running time reduction rate.  

As shown on the table, the realtime reduction rate is always lower than the theoretical flops reduction rate, which maybe due to the IO delay, buffer transfer corresponding to the hardware machine. Our customized convolution will cost additional running time for doing the tensor squeeze and expand operations, so the customized convolution realtime &darr; will be a little lower than the standard convolution realtime &darr;.

#### The model
We present the running time of the pruned model conresponds to the reference model, the reduction rates are shown as follows. In addition to the whole model, we also give the flops &darr; and realtime &darr; of the total pruned convolution layers, because we only prune the convolution layers in ResNet structures to obttain a sparse pruned model.

| Model flops &darr;  | Model running time &darr;  | Convolution layers flops &darr;  | Convolution layers running time &darr;  |
|---------------------|------------------------|----------------------------------|-------------------------------------|
|        26.83%       |         10.90%         |              27.95%              |                16.13%               |
|        41.37%       |         16.86%         |              43.10%              |                23.77%               |
|        54.87%       |         31.06%         |              57.16%              |                41.12%               |
|        66.05%       |         42.59%         |              68.80%              |                55.09%               |

As shown on the table, the convolution layers running time &darr; is lower than the theoretical convolution layers flops &darr;, the reason is similar to the single layer results. Moreover, due to the time cost in the BN, Relu and Fully-connected layers, the model running time &darr; is lower than convolution layers running time &darr;. In general, the running time reduction of the pruned convolution layers is consistent with the theoretical flops &darr;.

## Citation
@InProceedings{Li_2019_CVPR,
author = {Li, Tuanhui and Wu, Baoyuan and Yang, Yujiu and Fan, Yanbo and Zhang, Yong and Liu, Wei},
title = {Compressing Convolutional Neural Networks via Factorized Convolutional Filters},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2019}
}
