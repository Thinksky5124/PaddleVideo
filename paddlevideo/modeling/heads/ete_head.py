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

from paddle import ParamAttr
from paddle.nn import Linear
from paddle.regularizer import L2Decay
from .tsn_head import TSNHead
from ..registry import HEADS
from ..weight_init import weight_init_


@HEADS.register()
class ETEHead(TSNHead):

    def __init__(self,
                 num_classes,
                 in_channels,
                 sample_len,
                 sample_rate=2,
                 drop_ratio=0.5,
                 std=0.001,
                 data_format="NCHW",
                 **kwargs):
        super().__init__(num_classes,
                         in_channels,
                         drop_ratio=drop_ratio,
                         std=std,
                         data_format=data_format,
                         **kwargs)
        self.ce = nn.CrossEntropyLoss(ignore_index=-100)
        self.ce_soft = nn.CrossEntropyLoss(ignore_index=-100, soft_label=True)
        self.mse = nn.MSELoss(reduction='none')
        self.num_classes = num_classes
        self.sample_rate = sample_rate
        self.sample_len = sample_len

        self.fc = Linear(self.in_channels,
                         self.num_classes,
                         weight_attr=ParamAttr(learning_rate=5.0,
                                               regularizer=L2Decay(1e-4)),
                         bias_attr=ParamAttr(learning_rate=10.0,
                                             regularizer=L2Decay(0.0)))

        assert (data_format in [
            'NCHW', 'NHWC'
        ]), f"data_format must be 'NCHW' or 'NHWC', but got {data_format}"

        self.data_format = data_format

        self.stdv = std

        if self.sample_len % self.sample_rate != 0:
            raise NotImplementedError

        self.gather_index = paddle.zeros(shape=[self.sample_len], dtype='int32')
        repeat_gap = int(self.sample_len // self.sample_rate)
        for i in range(repeat_gap):
            for j in range(self.sample_rate):
                self.gather_index[i * self.sample_rate + j] = i + j * repeat_gap

    def init_weights(self):
        """Initiate the FC layer parameters"""
        weight_init_(self.fc, 'Normal', 'fc_0.w_0', 'fc_0.b_0', std=self.stdv)

    def forward(self, stage, feature, num_seg, mode):
        """MS-TCN no head
        """
        # stage shape [Stage_num N C T]
        stage_upsample = paddle.tile(stage,
                                     repeat_times=[1, 1, 1, self.sample_rate])
        stage_upsample = paddle.gather(stage_upsample,
                                       index=self.gather_index,
                                       axis=3)
        if mode in ['train', 'val']:
            if self.dropout is not None:
                x = self.dropout(feature)  # [N * num_seg, in_channels, 1, 1]

            if self.data_format == 'NCHW':
                x = paddle.reshape(x, x.shape[:2])
            else:
                x = paddle.reshape(x, x.shape[::3])
            score = self.fc(x)  # [N * num_seg, num_class]
            score = paddle.reshape(
                score, [-1, num_seg, score.shape[1]])  # [N, num_seg, num_class]
            score = paddle.mean(score, axis=1)  # [N, num_class]
            score = paddle.reshape(score,
                                   shape=[-1,
                                          self.num_classes])  # [N, num_class]
            # score = F.softmax(score)  #NOTE remove
            return stage_upsample, score
        else:
            return stage_upsample

    def feature_extract_loss(self, scores, video_gt):
        """calculate loss
        """
        ce_y = video_gt[:, ::self.sample_rate]
        ce_gt_onehot = F.one_hot(
            ce_y, num_classes=self.num_classes)  # shape [T, num_classes]
        smooth_label = paddle.sum(ce_gt_onehot, axis=1) / ce_gt_onehot.shape[1]
        ce_loss = self.ce_soft(scores, smooth_label)
        return ce_loss

    def segmentation_loss(self, output, video_gt):
        """calculate loss
        """
        # output shape [N C T]
        # video_gt shape [N T]
        ce_x = paddle.transpose(output, [0, 2, 1])  # shape [N T C]
        ce_y = video_gt
        loss = 0.0
        for batch_id in range(output.shape[0]):
            ce_loss = self.ce(ce_x[batch_id, :, :], ce_y[batch_id, :])
            loss = ce_loss

            mse = self.mse(
                F.log_softmax(output[batch_id, :, 1:], axis=1),
                F.log_softmax(output.detach()[batch_id, :, :-1], axis=1))
            mse = paddle.clip(mse, min=0, max=16)
            mse_loss = 0.15 * paddle.mean(mse)
            loss += mse_loss

        return loss
