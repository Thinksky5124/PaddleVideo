# Copyright (c) 2021  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from .base import BaseNeck
from ..registry import NECKS
import numpy as np
import copy
import random
import math

from paddle import ParamAttr
from ..weight_init import weight_init_

from ..backbones.ms_tcn import calculate_gain, KaimingUniform_like_torch
from ..backbones.ms_tcn import init_bias, SingleStageModel, DilatedResidualLayer


@NECKS.register()
class MSTCNNeck(BaseNeck):

    def __init__(self,
                 num_stages,
                 num_layers,
                 num_f_maps,
                 dim,
                 num_classes,
                 data_format="NCHW"):
        super().__init__()
        self.stage1 = SingleStageModel(num_layers, num_f_maps, dim, num_classes)
        self.stages = nn.LayerList([
            copy.deepcopy(
                SingleStageModel(num_layers, num_f_maps, num_classes,
                                 num_classes)) for s in range(num_stages - 1)
        ])

        assert (data_format in [
            'NCHW', 'NHWC'
        ]), f"data_format must be 'NCHW' or 'NHWC', but got {data_format}"

        self.data_format = data_format

    def forward(self, x, num_segs):
        """ MSTCN forward
        """
        results = {}
        # x.shape = [N * num_segs, in_channels, 7, 7]
        x = self.avgpool2d(x)  # [N * num_segs, in_channels, 1, 1]
        results['feature'] = x

        # x_seg = x.clone().detach()
        x = paddle.squeeze(x)  # [N * num_segs, in_channels]
        x = paddle.reshape(x, shape=[-1, num_segs,
                                     x.shape[-1]])  # [N, num_segs, in_channels]
        x_transpose = paddle.transpose(x,
                                       perm=[0, 2,
                                             1])  # [N, in_channels, num_segs]

        out = self.stage1(x_transpose)
        outputs = out.unsqueeze(0)
        for s in self.stages:
            out = s(F.softmax(out, axis=1))
            outputs = paddle.concat((outputs, out.unsqueeze(0)), axis=0)

        results['stage'] = outputs
        return results

    def init_weights(self):
        for layer in self.sublayers():
            if isinstance(layer, nn.Conv1D):
                layer.weight.set_value(
                    KaimingUniform_like_torch(layer.weight).astype('float32'))
                if layer.bias is not None:
                    layer.bias.set_value(
                        init_bias(layer.weight, layer.bias).astype('float32'))
                # weight_init_(layer, 'KaimingUniform')
                # weight_init_(layer, 'Normal', mean=0.0, std=0.02)

        self.avgpool2d = nn.AdaptiveAvgPool2D((1, 1),
                                              data_format=self.data_format)
