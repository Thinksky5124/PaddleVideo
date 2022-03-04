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

import numpy as np
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

        self.num_segs = backbone.num_seg
        self.sliding_strike = neck.sliding_strike
        self.clip_buffer_num = neck.clip_buffer_num
        self.sample_rate = head.sample_rate
        self.num_classes = head.num_classes

        self.memery_buffer = None

    def forward_net(self, imgs, start_frame, end_frame, mode):
        # step 2 extract feature
        """Define how the model is going to train, from input to output.
        """
        # imgs.shape=[N,T,C,H,W], for most commonly case
        imgs = paddle.reshape_(imgs, [-1] + list(imgs.shape[2:]))

        if self.backbone is not None:
            feature = self.backbone(imgs)
        else:
            feature = None

        # step 3 extract memery feature
        if self.neck is not None:
            seg_feature, cls_feature, memery_buffer = self.neck(
                feature, self.memery_buffer, self.num_segs, start_frame, end_frame)
        else:
            seg_feature = None
            cls_feature = None

        # step 4 store memery buffer
        self.memery_buffer = memery_buffer

        # step 5 segmentation
        if self.head is not None:
            headoutputs = self.head(seg_feature, cls_feature, self.num_segs, mode)
        else:
            headoutputs = None

        return headoutputs

    def train_step(self, data_batch):
        """Training step.
        """
        imgs_np = data_batch[0].numpy()
        video_gt = data_batch[1]

        # initilaze output result
        pred_score = np.zeros((imgs_np.shape[0], self.num_classes, imgs_np.shape[1] * self.sample_rate))

        total_seg_loss = 0.
        total_cls_loss = 0.
        total_top1 = 0.
        total_top5 = 0.
        sliding_cnt = 0
        # sliding segmentation
        for start_frame in range(0, imgs_np.shape[1], self.sliding_strike):
            # step 1 sliding sample
            end_frame = start_frame + self.num_segs
            if end_frame > imgs_np.shape[1]:
                end_frame = imgs_np.shape[1]
            imgs = paddle.to_tensor(imgs_np[:, start_frame:end_frame, :])
            clip_gt = video_gt[:, start_frame:end_frame]

            # call forward
            headoutputs = self.forward_net(imgs, start_frame, end_frame, 'train')

            output, scores = headoutputs
            # step 6 post precessing
            refine_start_frame = start_frame - (self.clip_buffer_num * self.num_segs)
            if refine_start_frame < 0:
                refine_start_frame = 0
            valid_len = end_frame - refine_start_frame
            pred_score[:, :, refine_start_frame:end_frame] = output[-1, :, :, -valid_len:].clone().detach()

            seg_loss = 0.
            for i in range(len(output)):
                seg_loss += self.head.segmentation_loss(output[i], clip_gt)

            cls_loss = self.head.feature_extract_loss(scores, clip_gt)

            top1, top5 = self.head.get_top_one_acc(scores, clip_gt)

            # log loss
            total_seg_loss = total_seg_loss + seg_loss
            total_cls_loss = total_cls_loss + cls_loss
            total_top1 = total_top1 + top1
            total_top5 = total_top5 + top5
            sliding_cnt = sliding_cnt + 1
        
        # mean loss
        total_cls_loss = total_cls_loss / sliding_cnt
        total_seg_loss = total_seg_loss / sliding_cnt
        total_top1 = total_top1 / sliding_cnt
        total_top5 = total_top5 / sliding_cnt
        loss = self.seg_weight * total_seg_loss + self.cls_weight * total_cls_loss

        # step last clear memery buffer
        self.memery_buffer = None


        predicted = paddle.argmax(pred_score, axis=1)
        predicted = paddle.squeeze(predicted)

        loss_metrics = dict()
        loss_metrics['loss'] = loss
        loss_metrics['loss_seg'] = total_seg_loss
        loss_metrics['loss_cls'] = total_cls_loss
        loss_metrics['F1@0.50'] = self.head.get_F1_score(predicted, video_gt)
        
        loss_metrics['top_1'] = total_top1
        loss_metrics['top_5'] = total_top5
        return loss_metrics

    def val_step(self, data_batch):
        """Validating setp.
        """
        imgs_np = data_batch[0].numpy()
        video_gt = data_batch[1]

        # initilaze output result
        pred_score = np.zeros((imgs_np.shape[0], self.num_classes, imgs_np.shape[1] * self.sample_rate))

        total_seg_loss = 0.
        total_cls_loss = 0.
        total_top1 = 0.
        total_top5 = 0.
        sliding_cnt = 0
        # sliding segmentation
        for start_frame in range(0, imgs_np.shape[1], self.sliding_strike):
            # step 1 sliding sample
            end_frame = start_frame + self.num_segs
            if end_frame > imgs_np.shape[1]:
                end_frame = imgs_np.shape[1]
            imgs = paddle.to_tensor(imgs_np[:, start_frame:end_frame, :])
            clip_gt = video_gt[:, start_frame:end_frame]

            # call forward
            headoutputs = self.forward_net(imgs, start_frame, end_frame, 'train')

            output, scores = headoutputs
            # step 6 post precessing
            refine_start_frame = start_frame - (self.clip_buffer_num * self.num_segs)
            if refine_start_frame < 0:
                refine_start_frame = 0
            valid_len = end_frame - refine_start_frame
            pred_score[:, :, refine_start_frame:end_frame] = output[-1, :, :, -valid_len:].clone().detach()

            seg_loss = 0.
            for i in range(len(output)):
                seg_loss += self.head.segmentation_loss(output[i], clip_gt)

            cls_loss = self.head.feature_extract_loss(scores, clip_gt)

            top1, top5 = self.head.get_top_one_acc(scores, clip_gt)

            # log loss
            total_seg_loss = total_seg_loss + seg_loss
            total_cls_loss = total_cls_loss + cls_loss
            total_top1 = total_top1 + top1
            total_top5 = total_top5 + top5
            sliding_cnt = sliding_cnt + 1
        
        # mean loss
        total_cls_loss = total_cls_loss / sliding_cnt
        total_seg_loss = total_seg_loss / sliding_cnt
        total_top1 = total_top1 / sliding_cnt
        total_top5 = total_top5 / sliding_cnt
        loss = self.seg_weight * total_seg_loss + self.cls_weight * total_cls_loss

        # step last clear memery buffer
        self.memery_buffer = None


        predicted = paddle.argmax(pred_score, axis=1)
        predicted = paddle.squeeze(predicted)

        loss_metrics = dict()
        loss_metrics['loss'] = loss
        loss_metrics['loss_seg'] = total_seg_loss
        loss_metrics['loss_cls'] = total_cls_loss
        loss_metrics['F1@0.50'] = self.head.get_F1_score(predicted, video_gt)
        
        loss_metrics['top_1'] = total_top1
        loss_metrics['top_5'] = total_top5
        return loss_metrics

    def test_step(self, data_batch):
        """Testing setp.
        """
        imgs_np = data_batch[0].numpy()

        # initilaze output result
        pred_score = np.zeros((imgs_np.shape[0], self.num_classes, imgs_np.shape[1] * self.sample_rate))

        # sliding segmentation
        for start_frame in range(0, imgs_np.shape[1], self.sliding_strike):
            # step 1 sliding sample
            end_frame = start_frame + self.num_segs
            if end_frame > imgs_np.shape[1]:
                end_frame = imgs_np.shape[1]
            imgs = paddle.to_tensor(imgs_np[:, start_frame:end_frame, :])

            # call forward
            headoutputs = self.forward_net(imgs, start_frame, end_frame, 'test')

            output, _ = headoutputs
            # step 6 post precessing
            refine_start_frame = start_frame - (self.clip_buffer_num * self.num_segs)
            if refine_start_frame < 0:
                refine_start_frame = 0
            valid_len = end_frame - refine_start_frame
            pred_score[:, :, refine_start_frame:end_frame] = output[-1, :, :, -valid_len:].clone().detach()

        # step last clear memery buffer
        self.memery_buffer = None


        predicted = paddle.argmax(pred_score, axis=1)
        predicted = paddle.squeeze(predicted)

        loss_metrics = dict()
        loss_metrics['predict'] = predicted
        loss_metrics['output_np'] = F.sigmoid(pred_score)
        return loss_metrics

    def infer_step(self, data_batch):
        """Infering setp.
        """
        imgs = data_batch[0]
        vid = data_batch[-1]

        # call forward
        headoutputs = self.forward_net(imgs, 'infer')
        output, _ = headoutputs
        predicted = paddle.argmax(output[-1], axis=1)
        predicted = paddle.squeeze(predicted)
        output_np = F.sigmoid(output[-1])
        return predicted, output_np
