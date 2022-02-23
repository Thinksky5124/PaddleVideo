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

from ...registry import SEGMENTERS
from .base import BaseSegmenter

import paddle
import paddle.nn.functional as F


@SEGMENTERS.register()
class ETE(BaseSegmenter):
    """MS-TCN model framework."""

    def __init__(self,
                 seg_weight=1.0,
                 cls_weight=1.0,
                 backbone=None,
                 neck=None,
                 head=None,
                 loss=None):
        super().__init__(backbone, neck, head, loss)
        self.seg_weight = seg_weight
        self.cls_weight = cls_weight

    def forward_net(self, imgs, mode):
        """Define how the model is going to train, from input to output.
        """
        # NOTE: As the num_segs is an attribute of dataset phase, and didn't pass to build_head phase, should obtain it from imgs(paddle.Tensor) now, then call self.head method.
        batch_size = imgs.shape[0]
        num_segs = imgs.shape[
            1]  # imgs.shape=[N,T,C,H,W], for most commonly case
        imgs = paddle.reshape_(imgs, [-1] + list(imgs.shape[2:]))

        if self.backbone is not None:
            feature = self.backbone(imgs)
        else:
            feature = None

        if self.neck is not None:
            neck_output = self.neck(feature, batch_size)
        else:
            neck_output = None

        if neck_output is not None:
            neck_output_feature = neck_output['feature']
            neck_output_stage = neck_output['stage']

        if self.head is not None:
            headoutputs = self.head(neck_output_stage, neck_output_feature,
                                    num_segs, mode)
        else:
            headoutputs = None

        return headoutputs

    def train_step(self, data_batch):
        """Training step.
        """
        imgs, video_gt, _ = data_batch

        # call forward
        headoutputs = self.forward_net(imgs, 'train')
        output, scores = headoutputs
        seg_loss = 0.
        for i in range(len(output)):
            seg_loss += self.head.segmentation_loss(output[i], video_gt)
        cls_loss = self.head.feature_extract_loss(scores, video_gt)
        loss = self.seg_weight * seg_loss + self.cls_weight * cls_loss

        predicted = paddle.argmax(output[-1], axis=1)
        predicted = paddle.squeeze(predicted)

        loss_metrics = dict()
        loss_metrics['loss'] = loss
        loss_metrics['F1@0.50'] = self.head.get_F1_score(predicted, video_gt)
        return loss_metrics

    def val_step(self, data_batch):
        """Validating setp.
        """
        imgs, video_gt, _ = data_batch

        # call forward
        headoutputs = self.forward_net(imgs, 'val')
        output, scores = headoutputs
        seg_loss = 0.
        for i in range(len(output)):
            seg_loss += self.head.segmentation_loss(output[i], video_gt)
        cls_loss = self.head.feature_extract_loss(scores, video_gt)
        loss = self.seg_weight * seg_loss + self.cls_weight * cls_loss

        predicted = paddle.argmax(output[-1], axis=1)
        predicted = paddle.squeeze(predicted)

        outputs_dict = dict()
        outputs_dict['loss'] = loss
        outputs_dict['F1@0.50'] = self.head.get_F1_score(predicted, video_gt)
        return outputs_dict

    def test_step(self, data_batch):
        """Testing setp.
        """
        imgs, _, _ = data_batch

        outputs_dict = dict()
        # call forward
        headoutputs = self.forward_net(imgs, 'test')
        output = headoutputs
        predicted = paddle.argmax(output[-1], axis=1)
        predicted = paddle.squeeze(predicted)
        outputs_dict['predict'] = predicted
        outputs_dict['output_np'] = F.sigmoid(output[-1])
        return outputs_dict

    def infer_step(self, data_batch):
        """Infering setp.
        """
        imgs = data_batch[0]

        # call forward
        headoutputs = self.forward_net(imgs, 'infer')
        output = headoutputs
        predicted = paddle.argmax(output[-1], axis=1)
        predicted = paddle.squeeze(predicted)
        output_np = F.sigmoid(output[-1])
        return predicted, output_np
