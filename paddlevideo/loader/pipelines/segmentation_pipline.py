#  Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import copy

import os
import numpy as np
import SimpleITK as sitk
from PIL import Image
import random
import paddle
from ..registry import PIPELINES
from .sample import Sampler
"""
pipeline ops for Action Segmentation Dataset.
"""


@PIPELINES.register()
class SegmentationSampler(object):

    def __init__(self, sample_rate):
        self.sample_rate = sample_rate

    def __call__(self, results):
        for key, data in results.items():
            if key not in ["video_name"]:
                if len(data.shape) == 1:
                    data = data[::self.sample_rate]
                    results[key] = copy.deepcopy(data)
                else:
                    data = data[:, ::self.sample_rate]
                    results[key] = copy.deepcopy(data)
        return results


@PIPELINES.register()
class VideoStramSampler(Sampler):
    """
    Sample frames id.
    NOTE: Use PIL to read image here, has diff with CV2
    Args:
        num_seg(int): number of segments.
        seg_len(int): number of sampled frames in each segment.
        valid_mode(bool): True or False.
        select_left: Whether to select the frame to the left in the middle when the sampling interval is even in the test mode.
    Returns:
        frames_idx: the index of sampled #frames.
    """

    def __init__(self,
                 sample_len,
                 seg_len,
                 sample_rate=2,
                 frame_interval=None,
                 valid_mode=False,
                 select_left=False,
                 dense_sample=False,
                 linspace_sample=False,
                 use_pil=True,
                 with_label=True):
        super(VideoStramSampler, self).__init__(sample_len // sample_len,
                                                seg_len,
                                                frame_interval=frame_interval,
                                                valid_mode=valid_mode,
                                                select_left=select_left,
                                                dense_sample=dense_sample,
                                                linspace_sample=linspace_sample,
                                                use_pil=use_pil)
        self.sample_rate = sample_rate
        self.sample_len = sample_len
        self.with_label = with_label

    def __call__(self, results):
        """
        Args:
            frames_len: length of frames.
        return:
            sampling id.
        """
        frames_len = int(results['frames_len'])

        start_frame = results['start_frame']
        end_frame = results['end_frame']
        frames_idx = []

        if start_frame > frames_len:
            start_frame = frames_len
        if end_frame > frames_len:
            end_frame = frames_len

        if results['format'] == 'video':
            frames_idx = list(range(start_frame, end_frame, self.sample_rate))
        else:
            raise NotImplementedError

        if self.with_label:
            classes = results['labels']
            labels = classes[start_frame:end_frame]
            results['labels'] = copy.deepcopy(labels)

        results = self._get(frames_idx, results)

        temporal_len = len(results['imgs'])
        if self.with_label:
            if temporal_len != self.sample_len // self.sample_rate or results[
                    'labels'].shape[0] != self.sample_len:
                imgbuf = np.zeros([3] + list(results['imgs'][0].size))
                imgs_pad = [Image.fromarray(imgbuf, mode='RGB')] * (
                    self.sample_len // self.sample_rate - temporal_len)
                imgs = results['imgs'] + imgs_pad
                labels_pad = np.full(self.sample_len -
                                     results['labels'].shape[0],
                                     -100,
                                     dtype='int64')
                labels = np.concatenate([results['labels'], labels_pad], axis=0)

                results['labels'] = labels
                results['imgs'] = imgs
        else:
            if temporal_len != self.sample_len // self.sample_rate:
                imgbuf = np.zeros([3] + list(results['imgs'][0].size))
                imgs_pad = [Image.fromarray(imgbuf, mode='RGB')] * (
                    self.sample_len // self.sample_rate - temporal_len)
                imgs = results['imgs'] + imgs_pad
                results['imgs'] = imgs
        return results
