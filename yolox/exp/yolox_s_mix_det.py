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
        self.num_classes = 1
        self.depth = 0.33
        self.width = 0.50
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.input_size = (608, 1088)
        self.test_size = (608, 1088)
        self.test_conf = 0.001
        self.nmsthre = 0.7