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
class ETENeck(BaseNeck):

    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers=2,
                 data_format="NCHW"):
        super().__init__()

        assert (data_format in [
            'NCHW', 'NHWC'
        ]), f"data_format must be 'NCHW' or 'NHWC', but got {data_format}"

        self.data_format = data_format
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.avgpool2d = nn.AdaptiveAvgPool2D((1, 1),
                                              data_format=self.data_format)
        # self.memery_unit = nn.LSTM(self.input_size,
        #                            self.hidden_size,
        #                            num_layers=self.num_layers)

    def forward(self, x, num_segs):
        """ ETEHead forward
        """
        # x.shape = [N * num_segs, in_channels, 7, 7]
        x = self.avgpool2d(x)
        # x.shape = [N * num_segs, in_channels, 1, 1]

        seg_x = paddle.squeeze(x)  # [N * num_segs, in_channels]
        seg_feature = paddle.reshape(seg_x,
                                     shape=[-1, num_segs, seg_x.shape[-1]
                                            ])  # [N, num_segs, in_channels]

        # # Todos: new video flash value
        # # Design: backward require
        # if self.h is None and self.c is None:
        #     # add pre information
        #     self.pad_zeros = paddle.zeros(
        #         [seg_x.shape[0], num_segs, self.hidden_size])
        #     seg_feature = paddle.concat([seg_x, self.pad_zeros], axis=2)
        #     # memeroy
        #     _, (h, c) = self.memery_unit(seg_x)
        #     self.h = h.detach()
        #     self.c = c.detach()
        # else:
        #     # concate pre information
        #     # N T D
        #     h_pad = paddle.tile(self.h[-1, :, :].unsqueeze(1),
        #                         repeat_times=[1, num_segs, 1])
        #     seg_feature = paddle.concat([seg_x, h_pad], axis=2)
        #     # memeroy
        #     _, (self.h, self.c) = self.memery_unit(seg_x, (self.h, self.c))

        seg_feature = paddle.transpose(seg_feature,
                                       perm=[0, 2,
                                             1])  # [N, in_channels, num_segs]

        return seg_feature, x

    def init_weights(self):
        # initalize h c
        self.h = None
        self.c = None
