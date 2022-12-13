import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
# from torch.backends import cudnn

from yolox.reidmodel.osnet import osnet_x1_0
from yolox.reidmodel.osent_PCB import osnet_pcb_x1_0
from yolox.reidmodel.resnest import resnest50
from yolox.reidmodel.shufflenetv2 import shufflenet_v2_x2_0
from yolox.deepsort_tracker.reid_model import ckpt


def postprocess(features):
    # Normalize feature to compute cosine distance
    features = F.normalize(features)
    features = features.cpu().data.numpy()
    return features


def preprocess(image, input_size):
    if len(image.shape) == 3:
        padded_img = np.ones((input_size[1], input_size[0], 3), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(input_size) * 114
    img = np.array(image)
    r = min(input_size[1] / img.shape[0], input_size[0] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    )
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    return padded_img, r

import logging



class FastReIDInterface:
    def __init__(self, model_type, weight_path, device, batch_size=8):
        super(FastReIDInterface, self).__init__()
        if device != 'cpu':
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        self.batch_size = batch_size
        self.model_type = model_type
        if self.model_type=="osnet_x1_0":
            self.model = osnet_x1_0(num_classes=751, pretrained=True, loss="triplet", inference=True, model_dir=weight_path)
        elif self.model_type=="osnet_pcb_x1_0":
            self.model = osnet_pcb_x1_0(num_classes=751, pretrained=True, loss="triplet", inference=True, model_dir=weight_path)
        elif self.model_type=='resnest50':
            self.model = resnest50(num_classes=751, pretrained=True, loss='triplet', use_gpu=True, weight_path=weight_path)
        elif self.model_type=='shufflenetv2_x2_0':
            self.model = shufflenet_v2_x2_0(num_classes=751, loss='triplet', pretrained=True, weight_path=weight_path)
        elif self.model_type=="ckpt":
            self.model = ckpt(num_class=751, reid=True, pretrained=True, loss="triplet", weight_path=weight_path)

        self.size = (64, 128)

        if self.device != 'cpu':
            self.model = self.model.eval().to(device='cuda').half()
        else:
            self.model = self.model.eval()

    def inference(self, image, detections):

        if detections is None or np.size(detections) == 0:
            return []

        H, W, _ = np.shape(image)    #得到图像的高宽

        batch_patches = []
        patches = []
        for d in range(np.size(detections, 0)):    #对检测一个个提取
            tlbr = detections[d, :4].astype(np.int_)   #得到检测框的位置
            # tlbr = detections[d, :4]
            tlbr[0] = max(0, tlbr[0])
            tlbr[1] = max(0, tlbr[1])
            tlbr[2] = min(W - 1, tlbr[2])
            tlbr[3] = min(H - 1, tlbr[3])    #整形成tlbr格式
            patch = image[tlbr[1]:tlbr[3], tlbr[0]:tlbr[2], :]  #切割形成patch

            # the model expects RGB inputs
            patch = patch[:, :, ::-1]

            patch, scale = preprocess(patch, self.size)


            patch = torch.as_tensor(patch.astype("float32").transpose(2, 0, 1))  #换图片通道
            patch = patch.to(device=self.device).half()  #半精度

            patches.append(patch)

            if (d + 1) % self.batch_size == 0:
                patches = torch.stack(patches, dim=0)
                batch_patches.append(patches)
                patches = []

        if len(patches):
            patches = torch.stack(patches, dim=0)
            batch_patches.append(patches)

        if self.model_type=="osnet_x1_0":
            features = np.zeros((0, 512))
        elif self.model_type=="osnet_pcb_x1_0":
            features = np.zeros((0, 1280))
        elif self.model_type=="ckpt":
            features = np.zeros((0, 512))
        elif self.model_type=='resnest50':
            features = np.zeros((0, 2048))
        elif self.model_type=='shufflenetv2_x2_0':
            features = np.zeros((0, 2048))



        for patches in batch_patches:

            pred = self.model(patches)
            pred[torch.isinf(pred)] = 1.0

            feat = postprocess(pred)

            features = np.vstack((features, feat))

        return features

