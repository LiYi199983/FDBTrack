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
        self.depth = 1.33
        self.width = 1.25
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.test_size = (800, 1440)
        self.test_conf = 0.1
        self.nmsthre = 0.7


