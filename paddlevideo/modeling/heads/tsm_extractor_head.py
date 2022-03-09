# Copyright (c) 2020  PaddlePaddle Authors. All Rights Reserved.
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
from ..registry import HEADS
from paddle.nn import AdaptiveAvgPool2D


@HEADS.register()
class TSMExtractorHead(nn.Layer):
    """ TSM Head

    Args:
        in_channels (int): The number of channles in input feature.
    """

    def __init__(self, in_channels, data_format="NCHW", **kwargs):
        super().__init__()

        assert (data_format in [
            'NCHW', 'NHWC'
        ]), f"data_format must be 'NCHW' or 'NHWC', but got {data_format}"

        self.data_format = data_format

    def init_weights(self):
        """Initiate the FC layer parameters"""
        self.avgpool2d = AdaptiveAvgPool2D((1, 1), data_format=self.data_format)

    def forward(self, x, num_seg):
        """Define how the tsm-head is going to run.

        Args:
            x (paddle.Tensor): The input data.
            num_segs (int): Number of segments.
        Returns:
            score: (paddle.Tensor) The classification scores for input samples.
        """
        # x.shape = [N * num_segs, in_channels, 7, 7]

        x = self.avgpool2d(x)  # [N * num_segs, in_channels, 1, 1]
        x = paddle.squeeze(x)
        return x
