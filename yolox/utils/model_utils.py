#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

from copy import deepcopy

import torch
import torch.nn as nn
from thop import profile

__all__ = [
    "get_model_info",
]

def get_model_info(model, tsize):

    stride = 64
    img = torch.zeros((1, 3, stride, stride), device=next(model.parameters()).device)
    flops, params = profile(deepcopy(model), inputs=(img,), verbose=False)
    params /= 1e6
    flops /= 1e9
    flops *= tsize[0] * tsize[1] / stride / stride * 2  # Gflops
    info = "Params: {:.2f}M, Gflops: {:.2f}".format(params, flops)
    return info



