"""
Code source: https://github.com/pytorch/vision
"""
from __future__ import division, absolute_import
import torch
import torch.utils.model_zoo as model_zoo
from torch import nn

__all__ = [
    'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0', 'shufflenet_v2_x1_5',
    'shufflenet_v2_x2_0'
]

model_urls = {
    'shufflenetv2_x0.5':
    'https://download.pytorch.org/models/shufflenetv2_x0.5-f707e7126e.pth',
    'shufflenetv2_x1.0':
    'https://download.pytorch.org/models/shufflenetv2_x1-5666bf0f80.pth',
    'shufflenetv2_x1.5': None,
    'shufflenetv2_x2.0': None,
}

model_dirs = {
    'shufflenetv2_x0.5':
    'G:\deep-person-reid-master\log\shufflenet_v2_x0_5\model\model.pth.tar-30',
    'shufflenetv2_x1.0': None,
    'shufflenetv2_x1.5': None,
    'shufflenetv2_x2.0': None,
}


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


class InvertedResidual(nn.Module):

    def __init__(self, inp, oup, stride):
        super(InvertedResidual, self).__init__()

        if not (1 <= stride <= 3):
            raise ValueError('illegal stride value')
        self.stride = stride

        branch_features = oup // 2
        assert (self.stride != 1) or (inp == branch_features << 1)

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                self.depthwise_conv(
                    inp, inp, kernel_size=3, stride=self.stride, padding=1
                ),
                nn.BatchNorm2d(inp),
                nn.Conv2d(
                    inp,
                    branch_features,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False
                ),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True),
            )

        self.branch2 = nn.Sequential(
            nn.Conv2d(
                inp if (self.stride > 1) else branch_features,
                branch_features,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False
            ),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            self.depthwise_conv(
                branch_features,
                branch_features,
                kernel_size=3,
                stride=self.stride,
                padding=1
            ),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(
                branch_features,
                branch_features,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False
            ),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def depthwise_conv(i, o, kernel_size, stride=1, padding=0, bias=False):
        return nn.Conv2d(
            i, o, kernel_size, stride, padding, bias=bias, groups=i
        )

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = channel_shuffle(out, 2)

        return out


class ShuffleNetV2(nn.Module):
    """ShuffleNetV2.
    
    Reference:
        Ma et al. ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design. ECCV 2018.

    Public keys:
        - ``shufflenet_v2_x0_5``: ShuffleNetV2 x0.5.
        - ``shufflenet_v2_x1_0``: ShuffleNetV2 x1.0.
        - ``shufflenet_v2_x1_5``: ShuffleNetV2 x1.5.
        - ``shufflenet_v2_x2_0``: ShuffleNetV2 x2.0.
    """

    def __init__(
        self, num_classes, loss, stages_repeats, stages_out_channels, **kwargs
    ):
        super(ShuffleNetV2, self).__init__()
        self.loss = loss

        if len(stages_repeats) != 3:
            raise ValueError(
                'expected stages_repeats as list of 3 positive ints'
            )
        if len(stages_out_channels) != 5:
            raise ValueError(
                'expected stages_out_channels as list of 5 positive ints'
            )
        self._stage_out_channels = stages_out_channels

        input_channels = 3
        output_channels = self._stage_out_channels[0]
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )
        input_channels = output_channels

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        stage_names = ['stage{}'.format(i) for i in [2, 3, 4]]
        for name, repeats, output_channels in zip(
            stage_names, stages_repeats, self._stage_out_channels[1:]
        ):
            seq = [InvertedResidual(input_channels, output_channels, 2)]
            for i in range(repeats - 1):
                seq.append(
                    InvertedResidual(output_channels, output_channels, 1)
                )
            setattr(self, name, nn.Sequential(*seq))
            input_channels = output_channels

        output_channels = self._stage_out_channels[-1]
        self.conv5 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )
        self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Linear(output_channels, num_classes)

    def featuremaps(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        return x

    def forward(self, x):
        f = self.featuremaps(x)
        v = self.global_avgpool(f)
        v = v.view(v.size(0), -1)

        if not self.training:
            return v

        y = self.classifier(v)

        if self.loss == 'softmax':
            return y
        elif self.loss == 'triplet':
            return y, v
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))



from yolox.reidmodel.osent_PCB import load_pretrained_weights


def shufflenet_v2_x0_5(num_classes, loss='softmax', pretrained=True, weight_path=None, **kwargs):
    model = ShuffleNetV2(
        num_classes, loss, [4, 8, 4], [24, 48, 96, 192, 1024], **kwargs
    )
    if pretrained:
        load_pretrained_weights(model, weight_path=weight_path)

    return model


def shufflenet_v2_x1_0(num_classes, loss='softmax', pretrained=True, weight_path=None, **kwargs):
    model = ShuffleNetV2(
        num_classes, loss, [4, 8, 4], [24, 116, 232, 464, 1024], **kwargs
    )
    if pretrained:
        load_pretrained_weights(model, weight_path=weight_path)
    return model


def shufflenet_v2_x1_5(num_classes, loss='softmax', pretrained=True, weight_path=None, **kwargs):
    model = ShuffleNetV2(
        num_classes, loss, [4, 8, 4], [24, 176, 352, 704, 1024], **kwargs
    )
    if pretrained:
        load_pretrained_weights(model, weight_path=weight_path)
    return model


def shufflenet_v2_x2_0(num_classes, loss='softmax', pretrained=True, weight_path=None, **kwargs):
    model = ShuffleNetV2(
        num_classes, loss, [4, 8, 4], [24, 244, 488, 976, 2048], **kwargs
    )
    if pretrained:
        load_pretrained_weights(model, weight_path=weight_path)
    return model

import numpy as np
import torchvision.transforms as transforms
import cv2

class SHUFFLENETExtractor(object):
    def __init__(self, model_path, use_cuda=True):
        self.model = shufflenet_v2_x2_0(num_classes=751, pretrained=True, loss="triplet", weight_path=model_path)  #turn extractor model
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

    model = shufflenet_v2_x1_0(num_classes=751, pretrained=False, loss="triplet")
    from torchsummary import summary
    from torchstat import stat
    input_size = (3, 128, 64)
    summary(model, input_size, device="cpu")
    stat(model, input_size)
