import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import os
import torch
import pdb
from torch.nn.utils.weight_norm import weight_norm

__all__ = ['ResNet','resnet34', 'resnet50']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, applySE=False):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        ## 여기에 self-attention 삽입
        if (applySE):
            return out, identity
        else:
            out += identity
            out = self.relu(out)

            return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, applySE=False):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        ## 여기에 self-attention 삽입
        if (applySE):
            return out, identity
        else:
            out += identity
            out = self.relu(out)

            return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

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

def resnet34(pretrained=False, root=os.path.join('save_model','resnet34-pretrained.pth'), **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model=ResNet(BasicBlock,[3,4,6,3],**kwargs)
    if pretrained:
        print('loading pretrained Res34 from {}'.format(root))
        res_pretrained = torch.load(root)
        model.load_state_dict(res_pretrained)
    return model


def resnet50(pretrained=False, root=os.path.join('save_model','resnet50-pretrained.pth'), **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        print('loading pretrained Res50 from {}'.format(root))
        res_pretrained=torch.load(root)
        model.load_state_dict(res_pretrained)
    return model

class Squeeze_N_Extension(nn.Module):
    def __init__(self, ResNet, reduction_ratio=16, C_list_for_SE=[]):
        super(Squeeze_N_Extension,self).__init__()

        self.ResNet=ResNet
        self.reduction_ratio=reduction_ratio
        self.relu=nn.ReLU(inplace=True)
        self.sigmoid=nn.Sigmoid()
        self.C_list=[]
        for featureDepth in C_list_for_SE:
            self.C_list.append( [weight_norm(nn.Linear(featureDepth, round(featureDepth/reduction_ratio) ) ), weight_norm(nn.Linear(round(featureDepth / reduction_ratio), featureDepth)) ])


    def forward(self, x, target_layer):

        layer_num = 0
        block_idx=0
        for layer in self.ResNet.children():
            #print('processing layer number : {}'.format(layer_num))
            if layer_num in target_layer:
                for sub_layer in layer.children():
                    x, identity=sub_layer(x, applySE=True)
                    att_x=torch.reshape(x, (x.size(0), x.size(1), -1))
                    att_x=torch.mean(att_x,2)
                    att_x=self.C_list[block_idx][0](att_x)
                    att_x=self.relu(att_x)
                    att_x=self.C_list[block_idx][1](att_x)
                    att_x=self.sigmoid(att_x)
                    att_x=att_x.unsqueeze(2).unsqueeze(3)
                    x=x*att_x
                    x += identity
                    x=self.relu(x)

                    block_idx += 1
            else:
                if(isinstance(layer, nn.Linear)):
                    x=x.squeeze()
                x=layer(x) # keep forwarding if not target layer to apply self-attention
            layer_num += 1
        return x


class Bottleneck_Attention_M(nn.Module):
    def __init__(self, ResNet, reduction_ratio=16, C_list_for_SE=[]):
        super(Bottleneck_Attention_M,self).__init__()

        self.ResNet=ResNet
        self.reduction_ratio=reduction_ratio
        self.relu=nn.ReLU(inplace=True)
        self.sigmoid=nn.Sigmoid()
        #self.bn_channel=nn.BatchNorm1d()
        self.bn_channel=[]
        self.bn_spatial=[]
        self.C_list=[] # channel attention
        self.S_list=[] # spatial attention
        for featureDepth in C_list_for_SE:
            reduced_feature_depth=round(featureDepth/reduction_ratio)
            self.C_list.append( [weight_norm(nn.Linear(featureDepth, reduced_feature_depth ) ), weight_norm(nn.Linear(reduced_feature_depth, featureDepth)) ])
            self.bn_channel.append(nn.BatchNorm1d(reduced_feature_depth, momentum=0.01))
            self.S_list.append([nn.Conv2d(featureDepth,reduced_feature_depth,kernel_size=1) , nn.Conv2d(reduced_feature_depth,reduced_feature_depth,kernel_size=3, dilation=4, padding=4),\
                                nn.Conv2d(reduced_feature_depth,reduced_feature_depth,kernel_size=3, dilation=4, padding=4), nn.Conv2d(reduced_feature_depth,1,kernel_size=1)])
            self.bn_spatial.append([nn.BatchNorm2d(reduced_feature_depth), nn.BatchNorm2d(reduced_feature_depth), nn.BatchNorm2d(reduced_feature_depth) ])

    def forward(self, x, target_layer):

        layer_num = 0
        block_idx=0
        for layer in self.ResNet.children():
            #print('processing layer number : {}'.format(layer_num))
            if layer_num in target_layer:
                for sub_layer in layer.children():
                    x=sub_layer(x) # we use output of block for BAM
                    c_att_x=torch.reshape(x, (x.size(0), x.size(1), -1))
                    c_att_x=torch.mean(c_att_x,2)
                    c_att_x=self.C_list[block_idx][0](c_att_x)
                    c_att_x = self.bn_channel[block_idx](c_att_x)
                    c_att_x=self.relu(c_att_x)
                    c_att_x=self.C_list[block_idx][1](c_att_x)
                    c_att_x=c_att_x.unsqueeze(2).unsqueeze(3)

                    s_att_x=x
                    for i in range(3):
                        s_att_x=self.S_list[block_idx][i](s_att_x)
                        s_att_x=self.bn_spatial[block_idx][i](s_att_x)
                        s_att_x=self.relu(s_att_x)
                    s_att_x=self.S_list[block_idx][3](s_att_x)

                    att_x=1+self.sigmoid(c_att_x*s_att_x)

                    x=x*att_x

                    block_idx += 1
            else:
                if(isinstance(layer, nn.Linear)):
                    x=x.squeeze()

                x=layer(x) # keep forwarding if not target layer to apply self-attention
            layer_num += 1
        return x


class C_Block_Attention_M(nn.Module):
    def __init__(self, ResNet, reduction_ratio=16, C_list_for_SE=[]):
        super(C_Block_Attention_M,self).__init__()

        self.ResNet=ResNet
        self.reduction_ratio=reduction_ratio
        self.relu=nn.ReLU(inplace=True)
        self.sigmoid=nn.Sigmoid()
        #self.bn_channel=nn.BatchNorm1d()
        self.bn_channel=[]
        self.bn_spatial=[]
        self.C_list=[] # channel attention
        self.S_list=[] # spatial attention
        for featureDepth in C_list_for_SE:
            reduced_feature_depth=round(featureDepth/reduction_ratio)
            self.C_list.append( [weight_norm(nn.Linear(featureDepth, reduced_feature_depth ) ), weight_norm(nn.Linear(reduced_feature_depth, featureDepth)) ])
            self.bn_channel.append(nn.BatchNorm1d(reduced_feature_depth, momentum=0.01))
            self.S_list.append(nn.Conv2d(2,1,kernel_size=7, padding=3))
            self.bn_spatial.append(nn.BatchNorm2d(1))

    def forward(self, x, target_layer):

        layer_num = 0
        block_idx=0
        for layer in self.ResNet.children():
            #print('processing layer number : {}'.format(layer_num))
            if layer_num in target_layer:
                for sub_layer in layer.children():
                    x, identity=sub_layer(x, applySE=True) # we use output of block for BAM
                    c_att_x=torch.reshape(x, (x.size(0), x.size(1), -1))
                    batch_size=c_att_x.size(0)
                    c_att_x_avg=torch.mean(c_att_x,2)
                    c_att_x_max,_=torch.max(c_att_x,2)
                    c_att_x=torch.cat( (c_att_x_avg, c_att_x_max), dim=0 )
                    c_att_x=self.C_list[block_idx][0](c_att_x)
                    c_att_x = self.bn_channel[block_idx](c_att_x)
                    c_att_x=self.relu(c_att_x)
                    c_att_x=self.C_list[block_idx][1](c_att_x)
                    c_att_x=torch.reshape(c_att_x,(batch_size,2,-1))
                    c_att_x=torch.sum(c_att_x,1)
                    c_att_x=self.sigmoid(c_att_x)
                    c_att_x=c_att_x.unsqueeze(2).unsqueeze(3)
                    x=c_att_x*x

                    s_att_x=x
                    s_att_x_avg=torch.mean(s_att_x,1)
                    s_att_x_max,_=torch.max(s_att_x,1)
                    s_att_x=torch.cat((s_att_x_avg.unsqueeze(1),s_att_x_max.unsqueeze(1)),dim=1)
                    
                    s_att_x=self.S_list[block_idx](s_att_x)
                    s_att_x=self.bn_spatial[block_idx](s_att_x)
                    s_att_x=self.sigmoid(s_att_x)

                    x=x*s_att_x

                    x += identity
                    x = self.relu(x)

                    block_idx += 1
            else:
                if(isinstance(layer, nn.Linear)):
                    x=x.squeeze()

                x=layer(x) # keep forwarding if not target layer to apply self-attention
            layer_num += 1
        return x