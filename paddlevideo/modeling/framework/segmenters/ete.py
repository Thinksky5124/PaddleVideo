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

from this import d
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
                 backbone=None,
                 neck=None,
                 head=None,
                 loss=None):
        super().__init__(backbone, neck, head, loss)
        self.seg_weight = seg_weight

        self.num_segs = neck.num_segs
        self.sliding_strike = neck.sliding_strike
        self.clip_buffer_num = neck.clip_buffer_num
        self.sample_rate = head.sample_rate
        self.num_classes = head.num_classes

        self.memery_buffer = None
        self.mask_buffer = None

    def forward_net(self, imgs, seg_mask, start_frame):
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
            seg_feature, memery_buffer, mask_buffer = self.neck(
                feature, seg_mask[:, :, -(self.num_segs * self.sample_rate):][:, :, ::self.sample_rate],
                self.memery_buffer, self.mask_buffer, start_frame)
            
            # step 4 store memery buffer
            self.memery_buffer = memery_buffer
            self.mask_buffer = mask_buffer
        else:
            seg_feature = feature

        # step 5 segmentation
        if self.head is not None:
            seg_score = self.head(seg_feature, seg_mask)
        else:
            seg_score = None

        return seg_score
    
    def _post_processing(self, pred_score, video_gt):
        pred_score_list = []
        pred_cls_list = []

        for bs in range(pred_score.shape[0]):
            index = np.where(video_gt[bs, :].numpy() == -100)
            ignore_start = min(index[0])
            predicted = paddle.argmax(pred_score[bs, :, :ignore_start], axis=0)
            predicted = paddle.squeeze(predicted)
            pred_cls_list.append(predicted)
            pred_score_list.append(pred_score[bs, :, :ignore_start])

        return pred_score_list, pred_cls_list

    def train_step(self, data_batch, optimizer):
        """Training step.
        """
        imgs_np = data_batch[0].numpy()
        video_gt = data_batch[1]
        seg_mask = data_batch[2]

        # initilaze output result
        pred_score = paddle.zeros((imgs_np.shape[0], self.num_classes, imgs_np.shape[1] * self.sample_rate))

        total_seg_loss = 0.
        sliding_num = imgs_np.shape[1] // self.sliding_strike
        if imgs_np.shape[1] % self.sliding_strike != 0:
            sliding_num = sliding_num + 1
        sliding_cnt = 0
        # sliding segmentation
        for start_frame in range(0, imgs_np.shape[1], self.sliding_strike):
            segment_len = self.sample_rate * self.num_segs * (self.clip_buffer_num + 1)
            # step 1 sliding sample
            end_frame = start_frame + self.num_segs
            # jump end none
            if end_frame > imgs_np.shape[1]:
                break
            # generate index
            start_overwrite_frame = segment_len - (self.sliding_strike * sliding_cnt + self.num_segs) * self.sample_rate
            if start_overwrite_frame < 0:
                start_overwrite_frame = 0
            gt_start_frame = (start_frame - self.clip_buffer_num * self.num_segs) * self.sample_rate
            if gt_start_frame < 0:
                gt_start_frame = 0
            gt_end_frame = (start_frame + self.num_segs) * self.sample_rate

            # sampling imgs labels
            imgs = paddle.to_tensor(imgs_np[:, start_frame:end_frame, :])
            sample_seg_mask = paddle.tile(
                seg_mask[:, gt_start_frame:gt_end_frame].unsqueeze(1),
                repeat_times=[1, self.num_classes, 1])
            seg_gt = video_gt[:, gt_start_frame:gt_end_frame]
            if seg_gt.shape[1] < segment_len:
                seg_gt_pad = paddle.to_tensor(np.full((seg_gt.shape[0], segment_len - seg_gt.shape[1]), -100, dtype=np.int64))
                seg_gt = paddle.concat([seg_gt_pad, seg_gt], axis=1)
                mask_pad = paddle.to_tensor(np.zeros((sample_seg_mask.shape[0], sample_seg_mask.shape[1], segment_len - sample_seg_mask.shape[2]), dtype=np.float32))
                sample_seg_mask = paddle.concat([mask_pad, sample_seg_mask], axis=2)

            # call forward
            headoutputs = self.forward_net(imgs, sample_seg_mask, start_frame)

            output = headoutputs

            # step 6 post precessing
            pred_score[:, :, gt_start_frame:gt_end_frame] = output[-1, :, :, start_overwrite_frame:].clone().detach()

            seg_loss = 0.
            for i in range(len(output)):
                seg_loss += self.head.loss(output[i], seg_gt, sample_seg_mask)

            # 4.2 backward
            seg_loss.backward()
            # 4.3 minimize
            optimizer.step()
            optimizer.clear_grad()

            # log loss
            total_seg_loss = total_seg_loss + seg_loss.clone().detach()
            sliding_cnt = sliding_cnt + 1
        
        # mean loss
        total_seg_loss = total_seg_loss / sliding_cnt

        # step last clear memery buffer
        self.memery_buffer = None
        self.mask_buffer = None

        pred_score_list, pred_cls_list = self._post_processing(pred_score, video_gt)

        loss_metrics = dict()
        loss_metrics['loss'] = total_seg_loss
        loss_metrics['F1@0.50'] = self.head.get_F1_score(pred_cls_list, video_gt)

        loss_metrics['predict'] = pred_cls_list
        loss_metrics['output_np'] = pred_score_list
        return loss_metrics

    def val_step(self, data_batch):
        """Validating setp.
        """
        imgs_np = data_batch[0].numpy()
        video_gt = data_batch[1]
        seg_mask = data_batch[2]

        # initilaze output result
        pred_score = paddle.zeros((imgs_np.shape[0], self.num_classes, imgs_np.shape[1] * self.sample_rate))

        total_seg_loss = 0.
        sliding_num = imgs_np.shape[1] // self.sliding_strike
        if imgs_np.shape[1] % self.sliding_strike != 0:
            sliding_num = sliding_num + 1
        sliding_cnt = 0
        # sliding segmentation
        for start_frame in range(0, imgs_np.shape[1], self.sliding_strike):
            segment_len = self.sample_rate * self.num_segs * (self.clip_buffer_num + 1)
            # step 1 sliding sample
            end_frame = start_frame + self.num_segs
            # jump end none
            if end_frame > imgs_np.shape[1]:
                break
            # generate index
            start_overwrite_frame = segment_len - (self.sliding_strike * sliding_cnt + self.num_segs) * self.sample_rate
            if start_overwrite_frame < 0:
                start_overwrite_frame = 0
            gt_start_frame = (start_frame - self.clip_buffer_num * self.num_segs) * self.sample_rate
            if gt_start_frame < 0:
                gt_start_frame = 0
            gt_end_frame = (start_frame + self.num_segs) * self.sample_rate

            # sampling imgs labels
            imgs = paddle.to_tensor(imgs_np[:, start_frame:end_frame, :])
            sample_seg_mask = paddle.tile(
                seg_mask[:, gt_start_frame:gt_end_frame].unsqueeze(1),
                repeat_times=[1, self.num_classes, 1])
            seg_gt = video_gt[:, gt_start_frame:gt_end_frame]
            if seg_gt.shape[1] < segment_len:
                seg_gt_pad = paddle.to_tensor(np.full((seg_gt.shape[0], segment_len - seg_gt.shape[1]), -100, dtype=np.int64))
                seg_gt = paddle.concat([seg_gt_pad, seg_gt], axis=1)
                mask_pad = paddle.to_tensor(np.zeros((sample_seg_mask.shape[0], sample_seg_mask.shape[1], segment_len - sample_seg_mask.shape[2]), dtype=np.float32))
                sample_seg_mask = paddle.concat([mask_pad, sample_seg_mask], axis=2)

            # call forward
            headoutputs = self.forward_net(imgs, sample_seg_mask, start_frame)

            output = headoutputs

            # step 6 post precessing
            pred_score[:, :, gt_start_frame:gt_end_frame] = output[-1, :, :, start_overwrite_frame:].clone().detach()

            seg_loss = 0.
            for i in range(len(output)):
                seg_loss += self.head.loss(output[i], seg_gt, sample_seg_mask)

            # log loss
            total_seg_loss = total_seg_loss + seg_loss.clone().detach()
            sliding_cnt = sliding_cnt + 1
        
        # mean loss
        total_seg_loss = total_seg_loss / sliding_cnt

        # step last clear memery buffer
        self.memery_buffer = None
        self.mask_buffer = None

        pred_score_list, pred_cls_list = self._post_processing(pred_score, video_gt)

        loss_metrics = dict()
        loss_metrics['loss'] = total_seg_loss
        loss_metrics['F1@0.50'] = self.head.get_F1_score(pred_cls_list, video_gt)

        loss_metrics['predict'] = pred_cls_list
        loss_metrics['output_np'] = pred_score_list
        return loss_metrics

    def test_step(self, data_batch):
        """Testing setp.
        """
        imgs_np = data_batch[0].numpy()
        video_gt = data_batch[1]
        seg_mask = data_batch[2]

        # initilaze output result
        pred_score = paddle.zeros((imgs_np.shape[0], self.num_classes, imgs_np.shape[1] * self.sample_rate))

        sliding_num = imgs_np.shape[1] // self.sliding_strike
        if imgs_np.shape[1] % self.sliding_strike != 0:
            sliding_num = sliding_num + 1
        sliding_cnt = 0
        # sliding segmentation
        for start_frame in range(0, imgs_np.shape[1], self.sliding_strike):
            segment_len = self.sample_rate * self.num_segs * (self.clip_buffer_num + 1)
            # step 1 sliding sample
            end_frame = start_frame + self.num_segs
            # jump end none
            if end_frame > imgs_np.shape[1]:
                break
            # generate index
            start_overwrite_frame = segment_len - (self.sliding_strike * sliding_cnt + self.num_segs) * self.sample_rate
            if start_overwrite_frame < 0:
                start_overwrite_frame = 0
            gt_start_frame = (start_frame - self.clip_buffer_num * self.num_segs) * self.sample_rate
            if gt_start_frame < 0:
                gt_start_frame = 0
            gt_end_frame = (start_frame + self.num_segs) * self.sample_rate

            # sampling imgs labels
            imgs = paddle.to_tensor(imgs_np[:, start_frame:end_frame, :])
            sample_seg_mask = paddle.tile(
                seg_mask[:, gt_start_frame:gt_end_frame].unsqueeze(1),
                repeat_times=[1, self.num_classes, 1])
            seg_gt = video_gt[:, gt_start_frame:gt_end_frame]
            if seg_gt.shape[1] < segment_len:
                seg_gt_pad = paddle.to_tensor(np.full((seg_gt.shape[0], segment_len - seg_gt.shape[1]), -100, dtype=np.int64))
                seg_gt = paddle.concat([seg_gt_pad, seg_gt], axis=1)
                mask_pad = paddle.to_tensor(np.zeros((sample_seg_mask.shape[0], sample_seg_mask.shape[1], segment_len - sample_seg_mask.shape[2]), dtype=np.float32))
                sample_seg_mask = paddle.concat([mask_pad, sample_seg_mask], axis=2)

            # call forward
            headoutputs = self.forward_net(imgs, sample_seg_mask, start_frame, 'val')

            output = headoutputs

            # step 6 post precessing
            pred_score[:, :, gt_start_frame:gt_end_frame] = output[-1, :, :, start_overwrite_frame:].clone().detach()

        # step last clear memery buffer
        self.memery_buffer = None
        self.mask_buffer = None

        pred_score_list, pred_cls_list = self._post_processing(pred_score, video_gt)

        loss_metrics = dict()

        loss_metrics['predict'] = pred_cls_list
        loss_metrics['output_np'] = pred_score_list
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
