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
import numpy as np

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

        # cls score
        self.overlap = 0.5

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

    def get_F1_score(self, predicted, groundTruth):
        # cls score
        correct = 0
        total = 0
        edit = 0
        tp = 0
        fp = 0
        fn = 0

        ignore = np.where(groundTruth == -100)

        for batch_size in range(predicted.shape[0]):
            # reshape output
            if ignore[1].shape[0] > 0 and ignore[0][0] == batch_size:
                recog_content = list(
                    predicted[batch_size, :ignore[1][0]].numpy())
                gt_content = list(
                    groundTruth[batch_size, :ignore[1][0]].numpy())
            else:
                recog_content = list(predicted[batch_size, :].numpy())
                gt_content = list(groundTruth[batch_size, :].numpy())

            for i in range(len(gt_content)):
                total += 1

                if gt_content[i] == recog_content[i]:
                    correct += 1

            edit_num = self.edit_score(recog_content, gt_content)
            edit += edit_num

            tp1, fp1, fn1 = self.f_score(recog_content, gt_content,
                                         self.overlap)
            tp += tp1
            fp += fp1
            fn += fn1

        # cls metric
        if tp + fp > 0.0:
            precision = tp / float(tp + fp)
        else:
            precision = 1.0
        if fp + fn > 0.0:
            recall = tp / float(fp + fn)
        else:
            recall = 1.0

        if precision + recall > 0.0:
            f1 = 2.0 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0
        f1 = np.nan_to_num(f1)
        return f1

    def get_labels_start_end_time(self, frame_wise_labels):
        labels = []
        starts = []
        ends = []
        last_label = frame_wise_labels[0]
        labels.append(frame_wise_labels[0])
        starts.append(0)
        for i in range(len(frame_wise_labels)):
            if frame_wise_labels[i] != last_label:
                labels.append(frame_wise_labels[i])
                starts.append(i)
                ends.append(i)
                last_label = frame_wise_labels[i]
        ends.append(i + 1)
        return labels, starts, ends

    def levenstein(self, p, y, norm=False):
        m_row = len(p)
        n_col = len(y)
        D = np.zeros([m_row + 1, n_col + 1], np.float)
        for i in range(m_row + 1):
            D[i, 0] = i
        for i in range(n_col + 1):
            D[0, i] = i

        for j in range(1, n_col + 1):
            for i in range(1, m_row + 1):
                if y[j - 1] == p[i - 1]:
                    D[i, j] = D[i - 1, j - 1]
                else:
                    D[i, j] = min(D[i - 1, j] + 1, D[i, j - 1] + 1,
                                  D[i - 1, j - 1] + 1)

        if norm:
            score = (1 - D[-1, -1] / max(m_row, n_col)) * 100
        else:
            score = D[-1, -1]

        return score

    def edit_score(self, recognized, ground_truth, norm=True):
        P, _, _ = self.get_labels_start_end_time(recognized)
        Y, _, _ = self.get_labels_start_end_time(ground_truth)
        return self.levenstein(P, Y, norm)

    def f_score(self, recognized, ground_truth, overlap):
        p_label, p_start, p_end = self.get_labels_start_end_time(recognized)
        y_label, y_start, y_end = self.get_labels_start_end_time(ground_truth)

        tp = 0
        fp = 0

        hits = np.zeros(len(y_label))

        for j in range(len(p_label)):
            intersection = np.minimum(p_end[j], y_end) - np.maximum(
                p_start[j], y_start)
            union = np.maximum(p_end[j], y_end) - np.minimum(
                p_start[j], y_start)
            IoU = (1.0 * intersection / union) * (
                [p_label[j] == y_label[x] for x in range(len(y_label))])
            # Get the best scoring segment
            idx = np.array(IoU).argmax()

            if IoU[idx] >= overlap and not hits[idx]:
                tp += 1
                hits[idx] = 1
            else:
                fp += 1
        fn = len(y_label) - sum(hits)
        return float(tp), float(fp), float(fn)
