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

import numpy as np
import argparse
import pandas as pd

from .registry import METRIC
from .base import BaseMetric
from paddlevideo.utils import get_logger

from .segmentation_utils import get_labels_scores_start_end_time, get_labels_start_end_time
from .segmentation_utils import levenstein, edit_score, f_score, boundary_AR
from .segmentation_utils import wrapper_compute_average_precision

logger = get_logger("paddlevideo")


class BaseSegmentationMetric(BaseMetric):
    """
    Test for Video Segmentation based model.
    """

    def __init__(self,
                 data_size,
                 batch_size,
                 overlap,
                 actions_map_file_path,
                 log_interval=1,
                 tolerance=5,
                 boundary_threshold=0.7,
                 max_proposal=100,
                 tiou_thresholds=np.linspace(0.5, 0.95, 10)):
        """prepare for metrics
        """
        super().__init__(data_size, batch_size, log_interval)
        # actions dict generate
        file_ptr = open(actions_map_file_path, 'r')
        actions = file_ptr.read().split('\n')[:-1]
        file_ptr.close()
        self.actions_dict = dict()
        for a in actions:
            self.actions_dict[a.split()[1]] = int(a.split()[0])

        # cls score
        self.overlap = overlap
        self.overlap_len = len(overlap)

        self.cls_tp = np.zeros(self.overlap_len)
        self.cls_fp = np.zeros(self.overlap_len)
        self.cls_fn = np.zeros(self.overlap_len)
        self.total_correct = 0
        self.total_edit = 0
        self.total_frame = 0
        self.total_video = 0

        # boundary score
        self.max_proposal = max_proposal
        self.AR_at_AN = [[] for _ in range(max_proposal)]

        # localization score
        self.tiou_thresholds = tiou_thresholds
        self.pred_results_dict = {
            "video-id": [],
            "t_start": [],
            "t_end": [],
            "label": [],
            "score": []
        }
        self.gt_results_dict = {
            "video-id": [],
            "t_start": [],
            "t_end": [],
            "label": []
        }

    def _update_score(self, vid, recog_content, gt_content, pred_detection,
                      gt_detection):
        # cls score
        correct = 0
        total = 0
        edit = 0

        for i in range(len(gt_content)):
            total += 1
            #accumulate
            self.total_frame += 1

            if gt_content[i] == recog_content[i]:
                correct += 1
                #accumulate
                self.total_correct += 1

        edit_num = edit_score(recog_content, gt_content)
        edit += edit_num
        self.total_edit += edit_num

        for s in range(self.overlap_len):
            tp1, fp1, fn1 = f_score(recog_content, gt_content, self.overlap[s])

            # accumulate
            self.cls_tp[s] += tp1
            self.cls_fp[s] += fp1
            self.cls_fn[s] += fn1

        # accumulate
        self.total_video += 1

        # proposal score
        for AN in range(self.max_proposal):
            AR = boundary_AR(pred_detection,
                             gt_detection,
                             self.overlap,
                             max_proposal=(AN + 1))
            self.AR_at_AN[AN].append(AR)

        # localization score

        p_label, p_start, p_end, p_scores = pred_detection
        g_label, g_start, g_end, _ = gt_detection
        p_vid_list = vid * len(p_label)
        g_vid_list = vid * len(g_label)

        # collect
        self.pred_results_dict[
            "video-id"] = self.pred_results_dict["video-id"] + p_vid_list
        self.pred_results_dict[
            "t_start"] = self.pred_results_dict["t_start"] + p_start
        self.pred_results_dict[
            "t_end"] = self.pred_results_dict["t_end"] + p_end
        self.pred_results_dict[
            "label"] = self.pred_results_dict["label"] + p_label
        self.pred_results_dict[
            "score"] = self.pred_results_dict["score"] + p_scores

        self.gt_results_dict[
            "video-id"] = self.gt_results_dict["video-id"] + g_vid_list
        self.gt_results_dict[
            "t_start"] = self.gt_results_dict["t_start"] + g_start
        self.gt_results_dict["t_end"] = self.gt_results_dict["t_end"] + g_end
        self.gt_results_dict["label"] = self.gt_results_dict["label"] + g_label

    def _transform_model_result(self, outputs_np, gt_np, outputs_arr):
        recognition = []
        for i in range(outputs_np.shape[0]):
            recognition = np.concatenate((recognition, [
                list(self.actions_dict.keys())[list(
                    self.actions_dict.values()).index(outputs_np[i])]
            ]))
        recog_content = list(recognition)

        gt_content = []
        for i in range(gt_np.shape[0]):
            gt_content = np.concatenate((gt_content, [
                list(self.actions_dict.keys())[list(
                    self.actions_dict.values()).index(gt_np[i])]
            ]))
        gt_content = list(gt_content)

        pred_detection = get_labels_scores_start_end_time(
            outputs_arr, recog_content, self.actions_dict)
        gt_detection = get_labels_scores_start_end_time(
            np.ones(outputs_arr.shape), gt_content, self.actions_dict)

        return [recog_content, gt_content, pred_detection, gt_detection]

    def update(self, batch_id, data, outputs):
        """update metrics during each iter
        """
        raise NotImplementedError

    def _compute_metrics(self):
        # cls metric
        Acc = 100 * float(self.total_correct) / self.total_frame
        Edit = (1.0 * self.total_edit) / self.total_video
        Fscore = dict()
        for s in range(self.overlap_len):
            precision = self.cls_tp[s] / float(self.cls_tp[s] + self.cls_fp[s])
            recall = self.cls_tp[s] / float(self.cls_tp[s] + self.cls_fn[s])

            f1 = 2.0 * (precision * recall) / (precision + recall)

            f1 = np.nan_to_num(f1) * 100
            Fscore[self.overlap[s]] = f1

        # proposal metric
        proposal_AUC = np.array(self.AR_at_AN) * 100
        AUC = np.mean(proposal_AUC)
        AR_at_AN1 = np.mean(proposal_AUC[0, :])
        AR_at_AN5 = np.mean(proposal_AUC[4, :])
        AR_at_AN15 = np.mean(proposal_AUC[14, :])

        # localization metric
        prediction = pd.DataFrame(self.pred_results_dict)
        ground_truth = pd.DataFrame(self.gt_results_dict)

        ap = wrapper_compute_average_precision(prediction, ground_truth,
                                               self.tiou_thresholds,
                                               self.actions_dict)

        mAP = ap.mean(axis=1) * 100
        average_mAP = mAP.mean()

        # save metric
        metric_dict = dict()
        metric_dict['Acc'] = Acc
        metric_dict['Edit'] = Edit
        for s in range(len(self.overlap)):
            metric_dict['F1@{:0.2f}'.format(
                self.overlap[s])] = Fscore[self.overlap[s]]
        metric_dict['Auc'] = AUC
        metric_dict['AR@AN1'] = AR_at_AN1
        metric_dict['AR@AN5'] = AR_at_AN5
        metric_dict['AR@AN15'] = AR_at_AN15
        metric_dict['mAP@0.5'] = mAP[0]
        metric_dict['avg_mAP'] = average_mAP

        return metric_dict

    def _log_metrics(self, metric_dict):
        # log metric
        log_mertic_info = "dataset model performence: "
        # preds ensemble
        log_mertic_info += "Acc: {:.4f}, ".format(metric_dict['Acc'])
        log_mertic_info += 'Edit: {:.4f}, '.format(metric_dict['Edit'])
        for s in range(len(self.overlap)):
            log_mertic_info += 'F1@{:0.2f}: {:.4f}, '.format(
                self.overlap[s],
                metric_dict['F1@{:0.2f}'.format(self.overlap[s])])

        # boundary metric
        log_mertic_info += "Auc: {:.4f}, ".format(metric_dict['Auc'])
        log_mertic_info += "AR@AN1: {:.4f}, ".format(metric_dict['AR@AN1'])
        log_mertic_info += "AR@AN5: {:.4f}, ".format(metric_dict['AR@AN5'])
        log_mertic_info += "AR@AN15: {:.4f}, ".format(metric_dict['AR@AN15'])

        # localization metric
        log_mertic_info += "mAP@0.5: {:.4f}, ".format(metric_dict['mAP@0.5'])
        log_mertic_info += "avg_mAP: {:.4f}, ".format(metric_dict['avg_mAP'])
        logger.info(log_mertic_info)

    def _clear_for_next_epoch(self):
        # clear for next epoch
        # cls
        self.cls_tp = np.zeros(self.overlap_len)
        self.cls_fp = np.zeros(self.overlap_len)
        self.cls_fn = np.zeros(self.overlap_len)
        self.total_correct = 0
        self.total_edit = 0
        self.total_frame = 0
        self.total_video = 0
        # proposal
        self.AR_at_AN = [[] for _ in range(self.max_proposal)]
        # localization
        self.pred_results_dict = {
            "video-id": [],
            "t_start": [],
            "t_end": [],
            "label": [],
            "score": []
        }
        self.gt_results_dict = {
            "video-id": [],
            "t_start": [],
            "t_end": [],
            "label": []
        }

    def accumulate(self):
        """accumulate metrics when finished all iters.
        """
        raise NotImplementedError


@METRIC.register
class SegmentationMetric(BaseSegmentationMetric):
    """
    Test for Video Segmentation based model.
    """

    def __init__(self,
                 data_size,
                 batch_size,
                 overlap,
                 actions_map_file_path,
                 log_interval=1,
                 tolerance=5,
                 boundary_threshold=0.7,
                 max_proposal=100,
                 tiou_thresholds=np.linspace(0.5, 0.95, 10)):
        """prepare for metrics
        """
        super().__init__(data_size, batch_size, overlap, actions_map_file_path,
                         log_interval, tolerance, boundary_threshold,
                         max_proposal, tiou_thresholds)

    def update(self, batch_id, data, outputs):
        """update metrics during each iter
        """
        groundTruth = data[1]
        vid = data[-1]

        predicted = outputs['predict']
        output_np = outputs['output_np']

        outputs_np = predicted.numpy()
        outputs_arr = output_np.numpy()[0, :]
        gt_np = groundTruth.numpy()[0, :]

        result = self._transform_model_result(outputs_np, gt_np, outputs_arr)
        recog_content, gt_content, pred_detection, gt_detection = result
        self._update_score(vid, recog_content, gt_content, pred_detection,
                           gt_detection)

    def accumulate(self):
        """accumulate metrics when finished all iters.
        """
        metric_dict = self._compute_metrics()
        self._log_metrics(metric_dict)
        self._clear_for_next_epoch()

        return metric_dict


@METRIC.register
class StreamSegmentationMetric(BaseSegmentationMetric):
    """
    Test for Video Segmentation based model.
    """

    def __init__(self,
                 data_size,
                 batch_size,
                 overlap,
                 actions_map_file_path,
                 log_interval=1,
                 tolerance=5,
                 boundary_threshold=0.7,
                 max_proposal=100,
                 tiou_thresholds=np.linspace(0.5, 0.95, 10)):
        """prepare for metrics
        """
        super().__init__(data_size, batch_size, overlap, actions_map_file_path,
                         log_interval, tolerance, boundary_threshold,
                         max_proposal, tiou_thresholds)

        self.results_dict = {}

    def update(self, batch_id, data, outputs):
        """update metrics during each iter
        """
        groundTruth = data[1]
        vid = data[-1]

        # [N, T]
        predicted = outputs['predict']
        # [N, C, T]
        output_np = outputs['output_np']

        ignore = np.where(groundTruth == -100)

        for b_id in range(len(vid)):
            video_name = vid[b_id]
            # reshape output
            if ignore[1].shape[0] > 0 and ignore[0][0] == b_id:
                outputs_np = predicted.numpy()[b_id, :ignore[1][0]]
                outputs_arr = output_np.numpy()[b_id, :, :ignore[1][0]]
                gt_np = groundTruth.numpy()[b_id, :ignore[1][0]]
            else:
                outputs_np = predicted.numpy()[b_id, :]
                outputs_arr = output_np.numpy()[b_id, :]
                gt_np = groundTruth.numpy()[b_id, :]
            # log output
            if video_name not in self.results_dict.keys():
                self.results_dict[video_name] = {}
                self.results_dict[video_name]['gt'] = [gt_np]
                self.results_dict[video_name]['b_arr'] = [outputs_arr]
                self.results_dict[video_name]['cls'] = [outputs_np]
            else:
                self.results_dict[video_name]['gt'].append(gt_np)
                self.results_dict[video_name]['b_arr'].append(outputs_arr)
                self.results_dict[video_name]['cls'].append(outputs_np)

    def accumulate(self):
        """accumulate metrics when finished all iters.
        """
        for vid in self.results_dict.keys():
            outputs_np_list = self.results_dict[vid]['cls']
            gt_np_list = self.results_dict[vid]['gt']
            outputs_arr_list = self.results_dict[vid]['b_arr']
            outputs_np = np.concatenate(outputs_np_list, axis=0)
            gt_np = np.concatenate(gt_np_list, axis=0)
            outputs_arr = np.concatenate(outputs_arr_list, axis=1)
            result = self._transform_model_result(outputs_np, gt_np,
                                                  outputs_arr)
            recog_content, gt_content, pred_detection, gt_detection = result
            self._update_score([vid], recog_content, gt_content, pred_detection,
                               gt_detection)

        metric_dict = self._compute_metrics()
        self._log_metrics(metric_dict)
        self._clear_for_next_epoch()

        return metric_dict
