from __future__ import print_function

import argparse
import os

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from model import resnet34, resnet50, C_Block_Attention_M, Bottleneck_Attention_M, Squeeze_N_Extension
import pdb
import progressbar
import numpy as np
from utils import load_model, accuracy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 Evaluation')
parser.add_argument('--test_batch', default=1024, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--drop', '--dropout', default=0, type=float,
                    metavar='Dropout', help='Dropout ratio')
parser.add_argument('--log_step', type=int , default=100, help='step size for prining log info')
parser.add_argument('--model', type=str, default='res50', help='res50 or res34')
parser.add_argument('--type', type=str, default='SE', help='SE or BAM or CBAM or Baseline')
parser.add_argument('--model_path', type=str, default='None', help='SE or BAM or CBAM or BASELINE pretrained file path')

args = parser.parse_args()

def main():

    start_epoch = 0

    # Data
    print('Initializing Cifar100-Res50')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    dataloader = datasets.CIFAR100
    num_classes = 100


    testset = dataloader(root='./data', train=False, download=False, transform=transform_test)
    testloader = data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=0)

    if(args.model == 'res50'):
        model=resnet50(pretrained=True)
        target_layer=[4,5,6,7]
        depth_list=[256, 256, 256, 512, 512, 512, 512, 1024, 1024, 1024,1024, 1024,1024, 2048, 2048, 2048]
    elif(args.model == 'res34'):
        model = resnet34(pretrained=True)
        target_layer = [4, 5, 6, 7]
        depth_list = [64, 64,64,128,128,128,128,256,256,256,256,256, 256, 512,512,512]

    assert args.model_path is not None
    if (args.type == 'SE'):
        type='SqueezeExcite'
    elif(args.type == 'BAM'):
        type='Bottleneck_Attention_Module'
    elif(args.type == 'CBAM'):
        type='C_Block_Attention_Module'
    else:
        type='Baseline'
    args.model_path=os.path.join('save_model',type, args.model, args.model_path)

    if(args.type == 'SE'):
        SE_Adapter=Squeeze_N_Extension(model, C_list_for_SE=depth_list)
        SE_Adapter.to(device)
        SE_Adapter,_,_=load_model(args.model_path, SE_Adapter, type='SE')
        SE_Adapter.eval()
        for C_ in SE_Adapter.C_list:
            for i in range(2):
                C_[i].eval()
                C_[i].to(device)

        The_Adapter = SE_Adapter

    elif(args.type=='BAM'):
        BAM_Adapter=Bottleneck_Attention_M(model, C_list_for_SE=depth_list)
        BAM_Adapter.to(device)
        BAM_Adapter, _, _ = load_model(args.model_path, BAM_Adapter, type='BAM')
        BAM_Adapter.eval()
        for (C_, S_, bn_C, bn_S) in zip(BAM_Adapter.C_list, BAM_Adapter.S_list, BAM_Adapter.bn_channel, BAM_Adapter.bn_spatial):
            for i in range(2):
                C_[i].eval()
                C_[i].to(device)
            for i in range(4):
                S_[i].eval()
                S_[i].to(device)
            for i in range(3):
                bn_S[i].eval()
                bn_S[i].to(device)
            bn_C.eval()
            bn_C.to(device)

        The_Adapter = BAM_Adapter

    elif(args.type == 'CBAM'):
        CBAM_Adapter=C_Block_Attention_M(model, C_list_for_SE=depth_list)
        CBAM_Adapter.to(device)
        CBAM_Adapter, _, _ = load_model(args.model_path, CBAM_Adapter, type='CBAM')
        CBAM_Adapter.eval()
        for (C_, S_, bn_C, bn_S) in zip(CBAM_Adapter.C_list, CBAM_Adapter.S_list, CBAM_Adapter.bn_channel,
                                        CBAM_Adapter.bn_spatial):
            for i in range(2):
                C_[i].eval()
                C_[i].to(device)
            S_.eval()
            S_.to(device)
            bn_S.eval()
            bn_S.to(device)
            bn_C.eval()
            bn_C.to(device)

        The_Adapter = CBAM_Adapter

    elif(args.type == 'Baseline'):
        model.to(device)
        model_pre = torch.load(args.model_path)
        model.load_state_dict(model_pre['ResNet_state'])
        model.eval()

        The_Adapter = model

    N = len(testloader)
    bar = progressbar.ProgressBar(maxval=N).start()
    i_test = 0
    total_step=len(testloader)


    #print(BAM_Adapter)

    num_correct_samples_top1=0
    num_correct_samples_top5=0

    for batch_idx, (inputs, targets) in enumerate(testloader):
        bar.update(i_test)
        inputs, targets = inputs.to(device), targets.to(device)
        if(args.type == 'Baseline'):
            logits = The_Adapter(inputs)
        else:
            logits=The_Adapter(inputs, target_layer)

        accr_top1, correct_sample_top1= accuracy(logits, targets)
        num_correct_samples_top1 += correct_sample_top1

        accr_top5, correct_sample_top5 = accuracy(logits, targets, topk=(1,2,3,4,5,))
        num_correct_samples_top5 += correct_sample_top5

        i_test += 1

    num_samples=len(testset)*1.0
    print('============================================================================================')
    print('Top 1 error : {}'.format((num_samples-num_correct_samples_top1)*100.0/num_samples))
    print('Top 5 error : {}'.format((num_samples - num_correct_samples_top5)*100.0 / num_samples))


if __name__ == '__main__':
    main()
