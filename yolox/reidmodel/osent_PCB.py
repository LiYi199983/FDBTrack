from __future__ import division, absolute_import
import warnings
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import cv2
import logging
import torchvision.transforms as transforms


__all__ = [
    'osnet_pcb_x1_0', 'osnet_pcb_x0_75', 'osnet_pcb_x0_5', 'osnet_pcb_x0_25', 'osnet_pcb_ibn_x1_0'
]

pretrained_urls = {
    'osnet_pcb_x1_0':
        "G:\deep-person-reid-master\log\osnet_pcb_x1_0_triplet_cosine\model\model.pth.tar-150",
    'osnet_pcb_x0_75':
        'https://drive.google.com/uc?id=1uwA9fElHOk3ZogwbeY5GkLI6QPTX70Hq',
    'osnet_pcb_x0_5':
        'https://drive.google.com/uc?id=16DGLbZukvVYgINws8u8deSaOqjybZ83i',
    'osnet_pcb_x0_25':
        'https://drive.google.com/uc?id=1rb8UN5ZzPKRc_xvtHlyDh-cSz88YX9hs',
    'osnet_pcb_ibn_x1_0':
        'https://drive.google.com/uc?id=1sr90V6irlYYDd4_4ISU2iruoRG8J__6l',
}


##########
# Basic layers
##########
class ConvLayer(nn.Module):   #定于组合卷积层
    """Convolution layer (conv + bn + relu)."""

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            groups=1,
            IN=False
    ):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
            groups=groups
        )
        if IN:
            self.bn = nn.InstanceNorm2d(out_channels, affine=True)
        else:
            self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Conv1x1(nn.Module):   #1*1卷积
    """1x1 convolution + bn + relu."""

    def __init__(self, in_channels, out_channels, stride=1, groups=1):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            1,
            stride=stride,
            padding=0,
            bias=False,
            groups=groups
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Conv1x1Linear(nn.Module):   #输出用的1*1卷积，没有带激活层的
    """1x1 convolution + bn (w/o non-linearity)."""

    def __init__(self, in_channels, out_channels, stride=1):
        super(Conv1x1Linear, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, 1, stride=stride, padding=0, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class Conv3x3(nn.Module):  #定义3*3的卷积层
    """3x3 convolution + bn + relu."""

    def __init__(self, in_channels, out_channels, stride=1, groups=1):
        super(Conv3x3, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            3,
            stride=stride,
            padding=1,
            bias=False,
            groups=groups
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.hswish = nn.Hardswish(inplace=True)
        # self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.hswish(x)
        # x = self.relu(x)
        return x


class LightConv3x3(nn.Module):   #轻型3*3卷积
    """Lightweight 3x3 convolution.

    1x1 (linear) + dw 3x3 (nonlinear).
    """

    def __init__(self, in_channels, out_channels):
        super(LightConv3x3, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 1, stride=1, padding=0, bias=False
        )
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            3,
            stride=1,
            padding=1,
            bias=False,
            groups=out_channels
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.hswish = nn.Hardswish(inplace=True)
        # self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn(x)
        x = self.hswish(x)
        # x = self.relu(x)
        return x


class LightConv5x5(nn.Module):   #轻型3*3卷积
    """Lightweight 3x3 convolution.

    1x1 (linear) + dw 3x3 (nonlinear).
    """

    def __init__(self, in_channels, out_channels):
        super(LightConv5x5, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 1, stride=1, padding=0, bias=False
        )
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            5,
            stride=1,
            padding=2,
            bias=False,
            groups=out_channels
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.hswish = nn.Hardswish(inplace=True)
        # self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn(x)
        x = self.hswish(x)
        # x = self.relu(x)
        return x

class LightConv7x7(nn.Module):   #轻型3*3卷积
    """Lightweight 3x3 convolution.

    1x1 (linear) + dw 3x3 (nonlinear).
    """

    def __init__(self, in_channels, out_channels):
        super(LightConv7x7, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 1, stride=1, padding=0, bias=False
        )
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            7,
            stride=1,
            padding=3,
            bias=False,
            groups=out_channels
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.hswish = nn.Hardswish(inplace=True)
        # self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn(x)
        x = self.hswish(x)
        # x = self.relu(x)
        return x




##########
# Building blocks for omni-scale feature learning
##########
class ChannelGate(nn.Module):    #通道维度的AG门
    """A mini-network that generates channel-wise gates conditioned on input tensor."""

    def __init__(
            self,
            in_channels,
            num_gates=None,
            return_gates=False,
            gate_activation='sigmoid',   #门激活用的是sigmoid，这不就是cbcm
            reduction=16,   #压缩率
            layer_norm=False
    ):
        super(ChannelGate, self).__init__()
        if num_gates is None:
            num_gates = in_channels
        self.return_gates = return_gates
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(
            in_channels,
            in_channels // reduction,
            kernel_size=1,
            bias=True,
            padding=0
        )
        self.norm1 = None
        if layer_norm:
            self.norm1 = nn.LayerNorm((in_channels // reduction, 1, 1))
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(
            in_channels // reduction,
            num_gates,
            kernel_size=1,
            bias=True,
            padding=0
        )
        if gate_activation == 'sigmoid':
            self.gate_activation = nn.Sigmoid()
        elif gate_activation == 'relu':
            self.gate_activation = nn.ReLU(inplace=True)
        elif gate_activation == 'linear':
            self.gate_activation = None
        else:
            raise RuntimeError(
                "Unknown gate activation: {}".format(gate_activation)
            )

    def forward(self, x):
        input = x      #h*w*c
        x = self.global_avgpool(x)  #1*1*c
        x = self.fc1(x)   #1*1*c//r
        if self.norm1 is not None:
            x = self.norm1(x)
        x = self.relu(x)
        x = self.fc2(x)   #1*1*c
        if self.gate_activation is not None:
            x = self.gate_activation(x)
        if self.return_gates:
            return x
        return input * x    #h*w*c  就是一个通道注意力


class Channelattentionmodule(nn.Module):   #CAM
    def __init__(self, in_channel, r=0.5):
        super(Channelattentionmodule, self).__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.linear = nn.Sequential(
            nn.Conv2d(in_channel, int(in_channel*r), kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(in_channel*r), in_channel, kernel_size=1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self,input):
        max_pool = self.maxpool(input)
        avg_pool = self.avgpool(input)

        max_wight = self.linear(max_pool)
        avg_wight = self.linear(avg_pool)

        channel_wight = self.sigmoid(max_wight+avg_wight)

        out = input * channel_wight
        return out

class Spatialattentionmodule(nn.Module):   #SAM
    def __init__(self, k):
        super(Spatialattentionmodule, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=(k, k), bias=False, stride=1, padding=int((k-1)/2))
        self.sigmord = nn.Sigmoid()

    def forward(self, input):
        max_pool = torch.max(input,dim=1).values
        avg_pool = torch.mean(input,dim=1)

        max_pool = torch.unsqueeze(max_pool,dim=1)
        avg_pool = torch.unsqueeze(avg_pool,dim=1)

        attention = torch.cat((max_pool,avg_pool),dim=1)
        spatial_wight = self.conv1(attention)
        wight = self.sigmord(spatial_wight)

        out = input * wight
        return out


class OSBlock(nn.Module):
    """Omni-scale feature learning block."""

    def __init__(
            self,
            in_channels,
            out_channels,
            IN=False,
            bottleneck_reduction=2,   #瓶颈压缩率
            **kwargs
    ):
        super(OSBlock, self).__init__()
        mid_channels = out_channels // bottleneck_reduction   #中间层的通道数
        self.conv1 = Conv1x1(in_channels, mid_channels)    #降维压缩  h*w*c->h*w*c//4
        self.conv2a = LightConv3x3(mid_channels, mid_channels)   #h*w*c//4   field 3*3
        self.conv2b = nn.Sequential(
            LightConv5x5(mid_channels, mid_channels),
        )     #field  5*5
        self.conv2c = nn.Sequential(
            LightConv7x7(mid_channels, mid_channels),
        )    #field   7*7
        self.gate = ChannelGate(mid_channels)    #生成一个单门的注意力块，所有的分支贡献一个注意力块
        self.conv3 = Conv1x1Linear(mid_channels, out_channels)    #不带激活的1*1卷积，把中间维度的转换为输出维度
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = Conv1x1Linear(in_channels, out_channels)   #不带激活的
        self.IN = None
        if IN:
            self.IN = nn.InstanceNorm2d(out_channels, affine=True)

    def forward(self, x):
        identity = x   #旁路 h*w*c
        x1 = self.conv1(x)  #h*w*c//4
        x2a = self.conv2a(x1)
        x2b = self.conv2b(x1)
        x2c = self.conv2c(x1)
        x2 = self.gate(x2a) + self.gate(x2b) + self.gate(x2c) #居然不是cat？？？而是直接叠加，这样参数规模确实是小了
        x3 = self.conv3(x2)   #输出
        if self.downsample is not None:
            identity = self.downsample(identity)   #如果存在下采样，对旁路也要下采样
        out = x3 + identity
        if self.IN is not None:
            out = self.IN(out)
        return F.hardswish(out)
        # return F.relu(out)  #与正常的一个残差块是一样的用法


class DimReduceLayer(nn.Module):  #维度降低层

    def __init__(self, in_channels, out_channels, nonlinear):
        super(DimReduceLayer, self).__init__()
        layers = []
        layers.append(
            nn.Conv2d(
                in_channels, out_channels, 1, stride=1, padding=0, bias=False
            )
        )
        layers.append(nn.BatchNorm2d(out_channels))

        if nonlinear == 'relu':
            layers.append(nn.ReLU(inplace=True))
        elif nonlinear == 'leakyrelu':
            layers.append(nn.LeakyReLU(0.1))
        elif nonlinear == "hswish":
            layers.append(nn.Hardswish(inplace=True))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


##########
# Network architecture
##########
class OSNet(nn.Module):
    """Omni-Scale Network.

    Reference:
        - Zhou et al. Omni-Scale Feature Learning for Person Re-Identification. ICCV, 2019.
        - Zhou et al. Learning Generalisable Omni-Scale Representations
          for Person Re-Identification. TPAMI, 2021.  #2021年论文才发表出来
    """

    def __init__(
            self,
            num_classes,
            blocks,   #列表
            layers,
            channels,
            feature_dim=512,
            loss='softmax',
            IN=False,
            parts=4,
            reduce_dim=256,
            nonlinear = "hswish",
            inference = False,
            return_featuremaps=False,
            **kwargs
    ):
        super(OSNet, self).__init__()
        num_blocks = len(blocks)
        assert num_blocks == len(layers)
        assert num_blocks == len(channels) - 1
        self.loss = loss   #最后的loss格式
        self.parts = parts
        self.inference = inference
        # self.feature_dim = feature_dim   #特征维度默认是512
        # 128*64*3
        # convolutional backbone   #卷积骨干网络
        self.conv1 = ConvLayer(3, channels[0], 7, stride=2, padding=3, IN=IN)  #7*7卷积，2倍下采样   64*32*c0
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)    #两倍下采样  32*16*c0
        self.conv2 = self._make_layer(
            blocks[0],
            layers[0],
            channels[0],
            channels[1],
            reduce_spatial_size=True,
            IN=IN
        )    #  16*8*c1
        self.sam = Spatialattentionmodule(k=7)
        self.conv3 = self._make_layer(
            blocks[1],
            layers[1],
            channels[1],
            channels[2],
            reduce_spatial_size=True
        )   #8*4*c2
        self.cam = Channelattentionmodule(in_channel=channels[2])
        self.conv4 = self._make_layer(
            blocks[2],
            layers[2],
            channels[2],
            channels[3],
            reduce_spatial_size=False
        )   #8*4*c3
        self.conv5 = Conv1x1(channels[3], channels[3])   #聚合层  8*4*c3
        self.parts_avgpool = nn.AdaptiveAvgPool2d((self.parts,1))
        self.dropout = nn.Dropout(p=0.5)
        self.conv6 = DimReduceLayer(
            channels[3], reduce_dim, nonlinear=nonlinear
        )
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)    #   1*1*c3
        self.feature_dim = reduce_dim
        self.classifier = nn.ModuleList(
            [
                nn.Linear(self.feature_dim, num_classes)
                for _ in range(self.parts+1)
            ]
        )




        # fully connected layer
        # self.fc = self._construct_fc_layer(
        #     self.feature_dim, channels[3], dropout_p=None
        # )    # b*c3->b*feature——dim
        # # identity classification layer
        # self.classifier = nn.Linear(self.feature_dim, num_classes)   #

        self._init_params()

    def _make_layer(
            self,
            block,
            layer,
            in_channels,
            out_channels,
            reduce_spatial_size,    #下采样
            IN=False
    ):
        layers = []

        layers.append(block(in_channels, out_channels, IN=IN))  #根据信息重写一个列表
        for i in range(1, layer):
            layers.append(block(out_channels, out_channels, IN=IN))

        if reduce_spatial_size:
            layers.append(
                nn.Sequential(
                    Conv1x1(out_channels, out_channels),
                    nn.AvgPool2d(2, stride=2)
                )   #带下采样的话，就全局平均池化2
            )

        return nn.Sequential(*layers)   #组成一个layer

    def _construct_fc_layer(self, fc_dims, input_dim, dropout_p=None):
        if fc_dims is None or fc_dims < 0:
            self.feature_dim = input_dim
            return None

        if isinstance(fc_dims, int):
            fc_dims = [fc_dims]

        layers = []
        for dim in fc_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout_p is not None:
                layers.append(nn.Dropout(p=dropout_p))
            input_dim = dim

        self.feature_dim = fc_dims[-1]

        return nn.Sequential(*layers)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu'
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)



    def featuremaps(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.sam(x)
        x = self.conv3(x)
        x = self.cam(x)
        x = self.conv4(x)
        x = self.conv5(x)   #8倍下采样 c3
        return x

    def forward(self, x, return_featuremaps=False):
        f = self.featuremaps(x)
        v_g = self.parts_avgpool(f)   # 4*1*c3
        v_f = self.global_avgpool(f)  #1*1*c3


        v_g = torch.cat((v_g, v_f), dim=2)  #5*1*c3

        v_g = self.dropout(v_g)
        v_h = self.conv6(v_g)    #b*7*1*rediam   #1*1的减维度层  v_h=5*1*256

        if self.inference:
            v_h = F.normalize(v_h, p=2, dim=1)
            return v_h.view(v_h.size(0), -1)

        if not self.training:
            v_h = F.normalize(v_h, p=2, dim=1)
            return v_h.view(v_h.size(0), -1)

        y = []
        for i in range(self.parts+1):
            v_h_i = v_h[:, :, i, :]   #b*w*h*c
            v_h_i = v_h_i.view(v_h_i.size(0), -1)  #对整个的6*1*c的张量划分，
            y_i = self.classifier[i](v_h_i)  #对每一层分配到对应的线性层里
            y.append(y_i)  #最后的y按表格的形式输出

        if self.loss == 'softmax':
            return y
        elif self.loss == 'triplet':
            v_h = F.normalize(v_h, p=2, dim=1)   #三元组损失还要输出一张特征图
            return y, v_h.view(v_h.size(0), -1)  #把输出的特征图直接拉成向量
        elif self.loss == "circle":
            v_h = F.normalize(v_h, p=2, dim=1)
            v_h = torch.squeeze(v_h)
            return y, v_h
        else:
            raise KeyError('Unsupported loss: {}'.format(self.loss))

        if retrun_featuremaps:
            return f


def init_pretrained_weights(model, model_dir):
    """Initializes model with pretrained weights.

    Layers that don't match with pretrained layers in name or size are kept unchanged.
    """
    if model_dir is None:
        import warnings
        warnings.warn(
            'ImageNet pretrained weights are unavailable for this model'
        )
        return
    pretrain_dict = torch.load(model_dir)
    model_dict = model.state_dict()
    pretrain_dict = {
        k: v
        for k, v in pretrain_dict.items()
        if k in model_dict and model_dict[k].size() == v.size()
    }
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)




##########
# Instantiation
##########
def osnet_pcb_x1_0(num_classes=1000, pretrained=True, loss='softmax',inference=False,model_dir=None, **kwargs):
    # standard size (width x1.0)
    model = OSNet(
        num_classes,
        blocks=[OSBlock, OSBlock, OSBlock],
        layers=[2, 2, 2],
        channels=[64, 256, 384, 512],
        loss=loss,
        inference=inference,
        **kwargs
    )
    if pretrained:
        load_pretrained_weights(model, weight_path=model_dir)
        print("load wight")
    return model


def osnet_pcb_x0_75(num_classes=1000, pretrained=True, loss='softmax',inference=False, model_dir=None, **kwargs):
    # medium size (width x0.75)
    model = OSNet(
        num_classes,
        blocks=[OSBlock, OSBlock, OSBlock],
        layers=[2, 2, 2],
        channels=[48, 192, 288, 384],
        loss=loss,
        inference=inference,
        **kwargs
    )
    if pretrained:
        load_pretrained_weights(model, weight_path=model_dir)
        print("load wight")
    return model


def osnet_pcb_x0_5(num_classes=1000, pretrained=True, loss='softmax',inference=False, model_dir=None, **kwargs):
    # tiny size (width x0.5)
    model = OSNet(
        num_classes,
        blocks=[OSBlock, OSBlock, OSBlock],
        layers=[2, 2, 2],
        channels=[32, 128, 192, 256],
        loss=loss,
        inference=inference,
        **kwargs
    )
    if pretrained:
        load_pretrained_weights(model, weight_path=model_dir)
        print("load wight")
    return model


def osnet_pcb_x0_25(num_classes=1000, pretrained=True, loss='softmax', inference=False, model_dir=None, **kwargs):
    # very tiny size (width x0.25)
    model = OSNet(
        num_classes,
        blocks=[OSBlock, OSBlock, OSBlock],
        layers=[2, 2, 2],
        channels=[16, 64, 96, 128],
        loss=loss,
        inference=inference,
        **kwargs
    )
    if pretrained:
        load_pretrained_weights(model, weight_path=model_dir)
        print("load wight")
    return model


def osnet_pcb_ibn_x1_0(
        num_classes=1000, pretrained=True, loss='softmax', inference=False, model_dir=None, **kwargs
):
    # standard size (width x1.0) + IBN layer
    # Ref: Pan et al. Two at Once: Enhancing Learning and Generalization Capacities via IBN-Net. ECCV, 2018.
    model = OSNet(
        num_classes,
        blocks=[OSBlock, OSBlock, OSBlock],
        layers=[2, 2, 2],
        channels=[64, 256, 384, 512],
        loss=loss,
        parts=6,
        reduce_dim=256,
        nonlinear="relu",
        IN=True,
        inference=inference,
        **kwargs
    )
    if pretrained:
        load_pretrained_weights(model, weight_path=model_dir)
        print("load wight")
    return model

import pickle
import shutil
import os.path as osp
import warnings
from functools import partial
from collections import OrderedDict
import torch
import torch.nn as nn




def load_checkpoint(fpath):
    r"""Loads checkpoint.

    ``UnicodeDecodeError`` can be well handled, which means
    python2-saved files can be read from python3.

    Args:
        fpath (str): path to checkpoint.

    Returns:
        dict
    """
    if fpath is None:
        raise ValueError('File path is None')
    fpath = osp.abspath(osp.expanduser(fpath))
    if not osp.exists(fpath):
        raise FileNotFoundError('File is not found at "{}"'.format(fpath))
    map_location = None if torch.cuda.is_available() else 'cpu'
    try:
        checkpoint = torch.load(fpath, map_location=map_location)
    except UnicodeDecodeError:
        pickle.load = partial(pickle.load, encoding="latin1")
        pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
        checkpoint = torch.load(
            fpath, pickle_module=pickle, map_location=map_location
        )
    except Exception:
        print('Unable to load checkpoint from "{}"'.format(fpath))
        raise
    return checkpoint

def load_pretrained_weights(model, weight_path):
    r"""Loads pretrianed weights to model.

    Features::
        - Incompatible layers (unmatched in name or size) will be ignored.
        - Can automatically deal with keys containing "module.".

    Args:
        model (nn.Module): network model.
        weight_path (str): path to pretrained weights.
    """
    checkpoint = load_checkpoint(weight_path)
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    model_dict = model.state_dict()
    new_state_dict = OrderedDict()
    matched_layers, discarded_layers = [], []

    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:] # discard module.

        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v
            matched_layers.append(k)
        else:
            discarded_layers.append(k)

    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict)

    if len(matched_layers) == 0:
        warnings.warn(
            'The pretrained weights "{}" cannot be loaded, '
            'please check the key names manually '
            '(** ignored and continue **)'.format(weight_path)
        )
    else:
        print(
            'Successfully loaded pretrained weights from "{}"'.
            format(weight_path)
        )
        if len(discarded_layers) > 0:
            print(
                '** The following layers are discarded '
                'due to unmatched keys or layer size: {}'.
                format(discarded_layers)
            )


class MYExtractor(object):
    def __init__(self, model_path, use_cuda=True):
        self.model = osnet_pcb_x1_0(num_classes=751, pretrained=True, loss="triplet", inference=True, model_dir=model_path)  #turn extractor model
        self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        self.model.to(self.device)
        self.size = (64, 128)#整型以后的目标框内外观为1/2
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])#3通道归一化操作

    def _preprocess(self, im_crops):#预处理函数
        """
        TODO:
            1. to float with scale from 0 to 1
            2. resize to (64, 128) as Market1501 dataset did
            3. concatenate to a numpy array
            3. to torch Tensor
            4. normalize
        """
        def _resize(im, size):
            return cv2.resize(im.astype(np.float32)/255., size)#这里用的是opencv来整形的

        im_batch = torch.cat([self.norm(_resize(im, self.size)).unsqueeze(
            0) for im in im_crops], dim=0).float()
        return im_batch#把预处理过的所有框，组合成一个batch

    def __call__(self, im_crops):
        im_batch = self._preprocess(im_crops)
        with torch.no_grad():
            im_batch = im_batch.to(self.device)
            features = self.model(im_batch)
        return features.cpu().numpy()#把特征传回到cpu，并转化为np格式的数组


if __name__ == "__main__":

    model = osnet_pcb_x0_5(num_classes=751, pretrained=False, loss="triplet")
    from torchsummary import summary
    from torchstat import stat
    input_size = (3, 128, 64)
    summary(model, input_size, device="cpu")
    stat(model, input_size)





