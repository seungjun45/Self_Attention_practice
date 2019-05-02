from __future__ import print_function

import argparse
import os
import shutil
import time
import random

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from model import resnet34, resnet50, Squeeze_N_Extension
import pdb
import progressbar
import numpy as np
from utils import save_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 Training')
parser.add_argument('--num_epoch', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--train_batch', default=3076, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test_batch', default=100, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--drop', '--dropout', default=0, type=float,
                    metavar='Dropout', help='Dropout ratio')
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--log_step', type=int , default=100, help='step size for prining log info')
parser.add_argument('--model', type=str, default='res50', help='res50 or res34')

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

    trainset = dataloader(root='./data', train=True, download=True, transform=transform_train)
    trainloader = data.DataLoader(trainset, batch_size=args.train_batch, shuffle=True, num_workers=0)

    testset = dataloader(root='./data', train=False, download=False, transform=transform_test)
    testloader = data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=0)

    if(args.model == 'res50'):
        model=resnet50(pretrained=True)
        target_layer=[4,5,6,7]
        depth_list=[256, 256, 256, 512, 512, 512, 512, 1024, 1024, 1024,1024, 1024,1024, 2048, 2048, 2048]
    elif(args.model == 'res34'):
        model = resnet34(pretrained=True)
        target_layer = [4, 5, 6, 7]
        depth_list = [64, 64,64,128,128,128,128,256,256,256,256,256,512,512,512]


    SE_Adapter=Squeeze_N_Extension(model, C_list_for_SE=depth_list)

    params=list(model.parameters())


    SE_Adapter.to(device)

    for SE in SE_Adapter.C_list:
        SE[0].to(device)
        SE[1].to(device)
        params += list(SE[0].parameters())+list(SE[1].parameters())

    optimizer = torch.optim.Adam(params, lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()

    N = args.num_epoch * len(trainloader)
    bar = progressbar.ProgressBar(maxval=N).start()
    i_train = 0
    total_step=len(trainloader)

    save_directory=os.path.join('save_model','SqueezeExcite')
    if not os.path.exists(save_directory):
        os.makedirs(save_directory) #saving model directory

    for epoch in range(args.num_epoch):
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            bar.update(i_train)
            inputs, targets = inputs.to(device), targets.to(device)
            logits=SE_Adapter(inputs, target_layer)
            loss = criterion(logits,targets)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 0.25)
            optimizer.step()

            if batch_idx % args.log_step == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                      .format(epoch, args.num_epoch, batch_idx, total_step, loss.item(), np.exp(loss.item())))

            i_train += 1

        model_path = os.path.join(save_directory,'model-{}.pth'.format(epoch))
        prev_model_path = os.path.join(save_directory, 'model-{}.pth'.format(epoch-10))
        save_model(model_path,SE_Adapter.ResNet,SE_Adapter.C_list,epoch,optimizer=optimizer)
        if (os.path.exists(prev_model_path)): # keep track of only 10 recent learned params
            os.remove(prev_model_path)



if __name__ == '__main__':
    main()
