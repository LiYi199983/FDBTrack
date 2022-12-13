# encoding: utf-8
import os
import random
import torch
import torch.nn as nn
import torch.distributed as dist

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.num_classes = 1  #只找第一个目标，人
        self.depth = 0.67
        self.width = 0.75
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.test_size = (800, 1440)  #测试的大小（800,1440）
        self.test_conf = 0.001
        self.nmsthre = 0.7
