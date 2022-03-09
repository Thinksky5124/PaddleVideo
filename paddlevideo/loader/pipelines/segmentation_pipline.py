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
                 seg_len,
                 sample_rate=2,
                 frame_interval=None,
                 valid_mode=False,
                 select_left=False,
                 dense_sample=False,
                 linspace_sample=False,
                 use_pil=True):
        super(VideoStramSampler, self).__init__(seg_len,
                                                seg_len,
                                                frame_interval=frame_interval,
                                                valid_mode=valid_mode,
                                                select_left=select_left,
                                                dense_sample=dense_sample,
                                                linspace_sample=linspace_sample,
                                                use_pil=use_pil)
        self.sample_rate = sample_rate

    def __call__(self, results):
        """
        Args:
            frames_len: length of frames.
        return:
            sampling id.
        """
        frames_len = int(results['frames_len'])
        results['frames_len'] = frames_len

        frames_idx = []

        if results['format'] == 'video':
            frames_idx = list(range(0, frames_len, self.sample_rate))
        else:
            raise NotImplementedError

        results = self._get(frames_idx, results)
        return results

@PIPELINES.register()
class BatchCompose(object):
    def __init__(self, clip_seg_num=15,sample_rate=4):
        self.clip_seg_num = clip_seg_num
        self.sample_rate = sample_rate

    def __call__(self, batch):
        max_imgs_len = 0
        max_labels_len = 0
        for mini_batch in batch:
            if max_imgs_len < mini_batch[0].shape[0]:
                max_imgs_len = mini_batch[0].shape[0]
            if max_labels_len < mini_batch[1].shape[0]:
                max_labels_len = mini_batch[1].shape[0]

        max_imgs_len = max_imgs_len + (self.clip_seg_num - max_imgs_len % self.clip_seg_num)
        max_labels_len = max_labels_len + ((self.clip_seg_num * self.sample_rate) - max_labels_len % (self.clip_seg_num * self.sample_rate))

        # shape imgs and labels len
        for batch_id in range(len(batch)):
            mini_batch_list = []
            list(batch[batch_id])
            # imgs
            pad_imgs_len = max_imgs_len - batch[batch_id][0].shape[0]
            pad_imgs = np.zeros([pad_imgs_len] + list(batch[batch_id][0].shape[1:]), dtype=batch[batch_id][0].dtype)
            # pad_imgs = np.random.normal(size = [pad_imgs_len] + list(batch[batch_id][0].shape[1:])).astype(batch[batch_id][0].dtype)
            mini_batch_list.append(np.concatenate([batch[batch_id][0], pad_imgs], axis=0))
            # lables
            pad_labels_len = max_labels_len - batch[batch_id][1].shape[0]
            pad_labels = np.full([pad_labels_len] + list(batch[batch_id][1].shape[1:]), -100, dtype=batch[batch_id][1].dtype)
            labels = np.concatenate([batch[batch_id][1], pad_labels], axis=0)
            mini_batch_list.append(labels)
            # masks
            mask = labels != -100
            mask = mask.astype(np.float32)
            mini_batch_list.append(mask)
            # vid
            mini_batch_list.append(batch[batch_id][-1])
            batch[batch_id] = tuple(mini_batch_list)
        result_batch = copy.deepcopy(batch)
        return result_batch