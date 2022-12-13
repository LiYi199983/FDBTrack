import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import logging
import torchvision.transforms as transforms


class BasicBlock(nn.Module):
    def __init__(self, c_in, c_out, is_downsample=False):
        super(BasicBlock, self).__init__()
        self.is_downsample = is_downsample
        if is_downsample:
            self.conv1 = nn.Conv2d(
                c_in, c_out, 3, stride=2, padding=1, bias=False)
        else:
            self.conv1 = nn.Conv2d(
                c_in, c_out, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        if is_downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(c_in, c_out, 1, stride=2, bias=False),
                nn.BatchNorm2d(c_out)
            )
        elif c_in != c_out:
            self.downsample = nn.Sequential(
                nn.Conv2d(c_in, c_out, 1, stride=1, bias=False),
                nn.BatchNorm2d(c_out)
            )
            self.is_downsample = True

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.bn2(y)
        if self.is_downsample:
            x = self.downsample(x)
        return F.relu(x.add(y), True)


def make_layers(c_in, c_out, repeat_times, is_downsample=False):
    blocks = []
    for i in range(repeat_times):
        if i == 0:
            blocks += [BasicBlock(c_in, c_out, is_downsample=is_downsample), ]
        else:
            blocks += [BasicBlock(c_out, c_out), ]
    return nn.Sequential(*blocks)


class Net(nn.Module):
    def __init__(self, num_classes=751, reid=False, loss="softmax"):
        super(Net, self).__init__()
        # 3 128 64
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # nn.Conv2d(32,32,3,stride=1,padding=1),
            # nn.BatchNorm2d(32),
            # nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, padding=1),
        )
        # 32 64 32
        self.layer1 = make_layers(64, 64, 2, False)
        # 32 64 32
        self.layer2 = make_layers(64, 128, 2, True)
        # 64 32 16
        self.layer3 = make_layers(128, 256, 2, True)
        # 128 16 8
        self.layer4 = make_layers(256, 512, 2, True)
        # 256 8 4
        self.avgpool = nn.AvgPool2d((8, 4), 1)
        # 256 1 1
        self.reid = reid
        self.loss = loss
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(256, num_classes),
        )

    def featuremaps(self, x):
        x = self.conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


    def forward(self, x , return_featuremaps=False):
        f = self.featuremaps(x)
        if return_featuremaps:
            return f

        x = self.avgpool(f)
        x = x.view(x.size(0), -1)
        if not self.training:
            return x
        if self.reid:
            x = F.normalize(x, p=2, dim=0)
            return x
        y = self.classifier(x)
        if self.loss == 'softmax':
            return y
        elif self.loss == 'triplet':  #三元组损失还要输出一张特征图
            return y, x  #把输出的特征图直接拉成向量

from yolox.reidmodel.osent_PCB import load_pretrained_weights
import logging

def load_weight(model=None,use_gpu=True, model_path=None):
    model=model
    device = "cuda" if torch.cuda.is_available() and use_gpu else "cpu"
    state_dict = torch.load(model_path, map_location=torch.device(device))[
        'net_dict']  # 网络常数
    model.load_state_dict(state_dict)  # 装权重
    logger = logging.getLogger("root.tracker")  # 这个路径是保存这个网络权重的路径
    logger.info("Loading weights from {}... Done!".format(model_path))  # 显示是否装载成功的记录器

def ckpt(num_class=751, reid=False, pretrained=False,loss="softmax", weight_path=None, **kwargs):
    model = Net(num_classes=num_class, reid=reid, loss=loss)
    if pretrained:
        load_weight(model, model_path=weight_path)
        print("load weight")
    return model


class Extractor(object):
    def __init__(self, model_path, use_cuda=True):
        self.net = Net(reid=True)
        self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        state_dict = torch.load(model_path, map_location=torch.device(self.device))[
            'net_dict']#网络常数
        self.net.load_state_dict(state_dict)#装权重
        logger = logging.getLogger("root.tracker")#这个路径是保存这个网络权重的路径
        logger.info("Loading weights from {}... Done!".format(model_path))#显示是否装载成功的记录器
        self.net.to(self.device)
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
            features = self.net(im_batch)
        return features.cpu().numpy()#把特征传回到cpu，并转化为np格式的数组


if __name__ == "__main__":

    model = Net(num_classes=751, reid=False)
    from torchsummary import summary
    from torchstat import stat
    input_size = (3, 128, 64)
    summary(model, input_size, device="cpu")
    stat(model, input_size)