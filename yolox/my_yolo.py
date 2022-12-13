import sys
import os
from pathlib import Path
from yolox.exp import get_exp
from loguru import logger
from yolox.utils import get_model_info, postprocess
from yolox.data.data_augment import ValTransform
import torchvision
import time
import torch
import numpy as np
import cv2

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

# NMS算法
def postprocess(prediction, num_classes, conf_thre=0.7, nms_thre=0.45):
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    for i, image_pred in enumerate(prediction):

        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Get score and class with highest confidence
        class_conf, class_pred = torch.max(
            image_pred[:, 5: 5 + num_classes], 1, keepdim=True
        )

        conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()
        # _, conf_mask = torch.topk((image_pred[:, 4] * class_conf.squeeze()), 1000)
        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)
        detections = detections[conf_mask]
        if not detections.size(0):
            continue

        nms_out_index = torchvision.ops.batched_nms(
            detections[:, :4],
            detections[:, 4] * detections[:, 5],
            detections[:, 6],
            nms_thre,
        )
        detections = detections[nms_out_index]
        if output[i] is None:
            output[i] = detections
        else:
            output[i] = torch.cat((output[i], detections))

    return output


class YoloX(object):
    # YOLOX固定参数
    _defaults = {
        'data_path': ROOT / 'dataset',
        # 'device': 'gpu'
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, exp_file, weight_path):  #提供样例文件和权重的位置
        self.__dict__.update(self._defaults)
        # 创建网络并载入模型
        exp = get_exp(exp_file)
        self.num_classes = exp.num_classes
        self.confthres = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = exp.get_model()
        logger.info("Model Summary: {}".format(get_model_info(self.model, exp.test_size)))
        # if self.device == 'gpu':
        self.model.to(self.device)
        self.model.eval()
        logger.info("loading checkpoint")
        ckpt = torch.load(weight_path, map_location="cpu")
        # load the model state dict
        self.model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    def detect_bounding_box(self, img, im0s):

        ratio = min(self.test_size[0] / im0s.shape[0], self.test_size[1] / im0s.shape[1])
        with torch.no_grad():
            t0 = time.time()
            img = img.to(self.device)
            outputs = self.model(img)
            # NMS
            outputs = outputs.to('cpu')
            outputs = postprocess(outputs, self.num_classes, self.confthres,
                                  self.nmsthre)
        return outputs
