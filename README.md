# CNN-FCF
This is a Pytorch implementation of our paper "Compressing Convolutional Neural Networks via Factorized Convolutional Filters" published in CVPR 2019.

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

## Requirements
- Python 3.6
- PyTorch 0.3.1
- TorchVision 0.2.0
- [OSQP](https://osqp.org/docs/get_started/python.html)

## Inference checkpoint files
The trained models files can be found in [google drive](https://drive.google.com/drive/folders/1VGqpOhAGe9YQcyZTGbzitsLuELjQdsXW?usp=sharing), which can be used to reproduce the results of our paper.

## Training fcf models
#### Training CIFAR-10
```
python train_cifar_fcf.py --sparse-rate=0.25 --model='resnet20' --pretrained-model='./checkpoints/pretrain/resnet20_cifar_full.pkl' --checkpoint-name='./checkpoints/fcf/resnet20_sparse_025'

python train_cifar_fcf.py --sparse-rate=0.25 --model='resnet32' --pretrained-model='./checkpoints/pretrain/resnet32_cifar_full.pkl' --checkpoint-name='./checkpoints/fcf/resnet32_sparse_025'

python train_cifar_fcf.py --sparse-rate=0.25 --model='resnet56' --pretrained-model='./checkpoints/pretrain/resnet56_cifar_full.pkl' --checkpoint-name='./checkpoints/fcf/resnet56_sparse_025'

python train_cifar_fcf.py --sparse-rate=0.25 --model='resnet110' --pretrained-model='./checkpoints/pretrain/resnet110_cifar_full.pkl' --checkpoint-name='./checkpoints/fcf/resnet110_sparse_025'
```
#### Training ImageNet
```
python train_imagenet_fcf.py --sparse-rate=0.25 --model='resnet34' --pretrained-model='./checkpoints/pretrain/resnet34_full.pth' --checkpoint-name='./checkpoints/fcf/resnet34_sparse_025'

python train_imagenet_fcf.py --sparse-rate=0.31 --model='resnet50' --pretrained-model='./checkpoints/pretrain/resnet50_full.pth' --checkpoint-name='./checkpoints/fcf/resnet50_sparse_031'
```

## Finetuning
Due to the numerical reason, there are still small changes after optimization, so we usually use finetuning to recover the model performance.
#### Finetuning CIFAR-10
```
python finetune_cifar.py --model='resnet20' --fcf-checkpoint='./checkpoints/fcf/resnet20_sparse_025.pth.tar' --best-checkpoint='./checkpoints/inference/resnet20_finetune_best_025'

python finetune_cifar.py --model='resnet32' --fcf-checkpoint='./checkpoints/fcf/resnet32_sparse_025.pth.tar' --best-checkpoint='./checkpoints/inference/resnet32_finetune_best_025'

python finetune_cifar.py --model='resnet56' --fcf-checkpoint='./checkpoints/fcf/resnet56_sparse_025.pth.tar' --best-checkpoint='./checkpoints/inference/resnet56_finetune_best_025'

python finetune_cifar.py --model='resnet110' --fcf-checkpoint='./checkpoints/fcf/resnet110_sparse_025.pth.tar' --best-checkpoint='./checkpoints/inference/resnet110_finetune_best_025'
```
#### Finetuning ImageNet
```
python finetune_imagenet.py --model='resnet34' --fcf-checkpoint='./checkpoints/fcf/resnet34_sparse_025.pth.tar' --best-checkpoint='./checkpoints/inference/resnet34_finetune_best_same025'

python finetune_imagenet.py --model='resnet50' --fcf-checkpoint='./checkpoints/fcf/resnet50_sparse_025.pth.tar' --best-checkpoint='./checkpoints/inference/resnet50_finetune_best_same031'
```

## Inference

#### Reproduce the CIFAR-10 results in our paper
```
python inference_cifar.py --model='resnet20' --n=6 --finetune-model='./checkpoints/inference/resnet20_finetune_best_025.pth.tar'

python inference_cifar.py --model='resnet32' --n=10 --finetune-model='./checkpoints/inference/resnet32_finetune_best_025.pth.tar'

python inference_cifar.py --model='resnet56' --n=18 --finetune-model='./checkpoints/inference/resnet56_finetune_best_025.pth.tar'

python inference_cifar.py --model='resnet110' --n=36 --finetune-model='./checkpoints/inference/resnet110_finetune_best_025.pth.tar'
```

#### Reproduce the ImageNet results in our paper
```
python inference_imagenet_resnet34.py --model='resnet34' -finetune-model='./checkpoints/inference/finetune_resnet34_best_same015.pth.tar'

python inference_imagenet_resnet50.py --model='resnet50' --finetune-model='./checkpoints/inference/finetune_resnet50_best_same031.pth.tar'
```

## Running time analysis
We now analyze the realtime reduction rate of our method. Considering that the convolution operation of each filter on GPU is independently, and dozens of process are conducted in parallel, we can not get the realtime reduction rate on GPU. The following experiments are conducted on CPU with ResNet34. 

#### Single layer
We first present the single layer running time reduction rate, our customized convolution is composed by squeeze, conv, expand, we also give the proportion of these three operations in the customized convolution, respectively.

<table class="tg">
  <tr>
    <th class="tg-uys7" rowspan="2">Theoretical flops &darr;</th>
    <th class="tg-uys7" rowspan="2">Standard realtime &darr;</th>
    <th class="tg-uys7" rowspan="2">Customized realtime &darr;</th>
    <th class="tg-uys7" colspan="3">Customized convolution</th>
  </tr>
  <tr>
    <td class="tg-uys7">squeeze</td>
    <td class="tg-uys7">conv</td>
    <td class="tg-uys7">expand</td>
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
2. Standard realtime &darr; is denoted as the standard convolution running time reduction rate.  
3. Customized realtime &darr; is denoted as customized convolution running time reduction rate.  

As shown on the table, the realtime reduction rate is always lower than the theoretical flops reduction rate, which maybe due to the IO delay, buffer transfer corresponding to the hardware machine. Our customized convolution will cost additional running time for doing the tensor squeeze and expand operations, so the customized convolution realtime &darr; will be a little lower than the standard convolution realtime &darr;.

#### Model inference
We test the inference running time of the pruned sparse model, the results are shown as follows. In addition to the entire model, we give the flops &darr; and realtime &darr; of the total pruned convolution layers in the model, because we only prune the convolution layers to obttain a sparse model.

| Model flops &darr;  | Model realtime &darr;  | Convolution layers flops &darr;  | Convolution layers realtime &darr;  |
|---------------------|------------------------|----------------------------------|-------------------------------------|
|        26.83%       |         10.90%         |              27.95%              |                16.13%               |
|        41.37%       |         16.86%         |              43.10%              |                23.77%               |
|        54.87%       |         31.06%         |              57.16%              |                41.12%               |
|        66.05%       |         42.59%         |              68.80%              |                55.09%               |

As shown on the table, the convolution layers realtime &darr; is lower than the theoretical convolution layers flops &darr;, the reason is same as the single layer results. Due to the time cost in the BN, Relu and Fully-connected layers, the model realtime &darr; is lower than convolution layers realtime &darr;. In general, the realtime reduction of the pruned convolution layers is consistent with the theoretical flops &darr;.

