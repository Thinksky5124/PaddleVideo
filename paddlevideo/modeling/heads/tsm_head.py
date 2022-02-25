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

import math
import paddle
from paddle import ParamAttr
from paddle.nn import Linear
import paddle.nn.functional as F
from paddle.regularizer import L2Decay
from .tsn_head import TSNHead
from ..registry import HEADS

from ..weight_init import weight_init_


@HEADS.register()
class TSMHead(TSNHead):
    """ TSM Head

    Args:
        num_classes (int): The number of classes to be classified.
        in_channels (int): The number of channles in input feature.
        loss_cfg (dict): Config for building config. Default: dict(name='CrossEntropyLoss').
        drop_ratio(float): drop ratio. Default: 0.5.
        std(float): Std(Scale) value in normal initilizar. Default: 0.001.
        kwargs (dict, optional): Any keyword argument to initialize.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
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

    def init_weights(self):
        """Initiate the FC layer parameters"""
        weight_init_(self.fc, 'Normal', 'fc_0.w_0', 'fc_0.b_0', std=self.stdv)

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

        if self.dropout is not None:
            x = self.dropout(x)  # [N * num_seg, in_channels, 1, 1]

        if self.data_format == 'NCHW':
            x = paddle.reshape(x, x.shape[:2])
        else:
            x = paddle.reshape(x, x.shape[::3])
        score = self.fc(x)  # [N * num_seg, num_class]
        score = paddle.reshape(
            score, [-1, num_seg, score.shape[1]])  # [N, num_seg, num_class]
        score = paddle.mean(score, axis=1)  # [N, num_class]
        score = paddle.reshape(score,
                               shape=[-1, self.num_classes])  # [N, num_class]
        # score = F.softmax(score)  #NOTE remove
        return score


# @HEADS.register()
# class TSMHead(TSNHead):
#     """ TSM Head

#     Args:
#         num_classes (int): The number of classes to be classified.
#         in_channels (int): The number of channles in input feature.
#         loss_cfg (dict): Config for building config. Default: dict(name='CrossEntropyLoss').
#         drop_ratio(float): drop ratio. Default: 0.5.
#         std(float): Std(Scale) value in normal initilizar. Default: 0.001.
#         kwargs (dict, optional): Any keyword argument to initialize.
#     """
#     def __init__(self,
#                  num_classes,
#                  in_channels,
#                  drop_ratio=0.5,
#                  std=0.001,
#                  data_format="NCHW",
#                  **kwargs):
#         super().__init__(num_classes,
#                          in_channels,
#                          drop_ratio=drop_ratio,
#                          std=std,
#                          data_format=data_format,
#                          **kwargs)

#         self.fc = Linear(self.in_channels,
#                          self.num_classes,
#                          weight_attr=ParamAttr(learning_rate=5.0,
#                                                regularizer=L2Decay(1e-4)),
#                          bias_attr=ParamAttr(learning_rate=10.0,
#                                              regularizer=L2Decay(0.0)))

#         assert (data_format in [
#             'NCHW', 'NHWC'
#         ]), f"data_format must be 'NCHW' or 'NHWC', but got {data_format}"

#         self.data_format = data_format

#         self.stdv = std

#         self.sample_rate = 2
#         self.ce_soft = paddle.nn.CrossEntropyLoss(ignore_index=-100, soft_label=True)

#     def init_weights(self):
#         """Initiate the FC layer parameters"""
#         weight_init_(self.fc, 'Normal', 'fc_0.w_0', 'fc_0.b_0', std=self.stdv)

#     def forward(self, x, num_seg):
#         """Define how the tsm-head is going to run.

#         Args:
#             x (paddle.Tensor): The input data.
#             num_segs (int): Number of segments.
#         Returns:
#             score: (paddle.Tensor) The classification scores for input samples.
#         """
#         # x.shape = [N * num_segs, in_channels, 7, 7]

#         x = self.avgpool2d(x)  # [N * num_segs, in_channels, 1, 1]

#         if self.dropout is not None:
#             x = self.dropout(x)  # [N * num_seg, in_channels, 1, 1]

#         if self.data_format == 'NCHW':
#             x = paddle.reshape(x, x.shape[:2])
#         else:
#             x = paddle.reshape(x, x.shape[::3])
#         score = self.fc(x)  # [N * num_seg, num_class]
#         score = paddle.reshape(
#             score, [-1, num_seg, score.shape[1]])  # [N, num_seg, num_class]
#         score = paddle.mean(score, axis=1)  # [N, num_class]
#         score = paddle.reshape(score,
#                                shape=[-1, self.num_classes])  # [N, num_class]
#         # score = F.softmax(score)  #NOTE remove
#         return score

#     def loss(self, scores, labels, valid_mode=False, if_top5=True, **kwargs):
#         """Calculate the loss accroding to the model output ```scores```,
#            and the target ```labels```.

#         Args:
#             scores (paddle.Tensor): The output of the model.
#             labels (paddle.Tensor): The target output of the model.

#         Returns:
#             losses (dict): A dict containing field 'loss'(mandatory) and 'top1_acc', 'top5_acc'(optional).

#         """
#         labels_gt = labels[0]
#         losses = dict()
#         loss = self.feature_extract_loss(scores, labels_gt)

#         ce_y = labels_gt[:, ::self.sample_rate]
#         ce_gt_onehot = F.one_hot(
#             ce_y, num_classes=self.num_classes)  # shape [T, num_classes]
#         smooth_label = paddle.sum(ce_gt_onehot, axis=1) / ce_gt_onehot.shape[1]
#         labels = paddle.argmax(smooth_label, axis=1).unsqueeze(1)
#         if if_top5:
#             top1, top5 = self.get_acc(scores, labels, valid_mode)
#             losses['top1'] = top1
#             losses['top5'] = top5
#             losses['loss'] = loss
#         else:
#             top1 = self.get_acc(scores, labels, valid_mode, if_top5)
#             losses['top1'] = top1
#             losses['loss'] = loss
#         return losses

#     def feature_extract_loss(self, scores, video_gt):
#         """calculate loss
#         """
#         ce_y = video_gt[:, ::self.sample_rate]
#         ce_gt_onehot = F.one_hot(
#             ce_y, num_classes=self.num_classes)  # shape [T, num_classes]
#         smooth_label = paddle.sum(ce_gt_onehot, axis=1) / ce_gt_onehot.shape[1]
#         ce_loss = self.ce_soft(scores, smooth_label)
#         return ce_loss
