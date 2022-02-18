# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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

import os.path as osp
import copy
import random
import numpy as np
import os
import cv2

from ..registry import DATASETS
from .base import BaseDataset
from ...utils import get_logger

logger = get_logger("paddlevideo")


@DATASETS.register()
class SegmentationDataset(BaseDataset):
    """Video dataset for action recognition
       The dataset loads raw videos and apply specified transforms on them.
       The index file is a file with multiple lines, and each line indicates
       a sample video with the filepath and label, which are split with a whitesapce.
       Example of a inde file:
        file tree:
        ─── GTEA
            ├── Videos
            │   ├── S1_Cheese_C1.mp4
            │   ├── S1_Coffee_C1.mp4
            │   ├── S1_CofHoney_C1.mp4
            │   └── ...
            ├── groundTruth
            │   ├── S1_Cheese_C1.txt
            │   ├── S1_Coffee_C1.txt
            │   ├── S1_CofHoney_C1.txt
            │   └── ...
            ├── splits
            │   ├── test.split1.bundle
            │   ├── test.split2.bundle
            │   ├── test.split3.bundle
            │   └── ...
            └── mapping.txt
       Args:
           file_path(str): Path to the index file.
           pipeline(XXX): A sequence of data transforms.
           **kwargs: Keyword arguments for ```BaseDataset```.
    """

    def __init__(self,
                 file_path,
                 videos_path,
                 gt_path,
                 pipeline,
                 sample_len,
                 actions_map_file_path,
                 num_retries=5,
                 suffix='',
                 **kwargs):
        self.num_retries = num_retries
        self.suffix = suffix
        self.sample_len = sample_len
        self.videos_path = videos_path
        self.gt_path = gt_path
        self.actions_map_file_path = actions_map_file_path

        # actions dict generate
        file_ptr = open(self.actions_map_file_path, 'r')
        actions = file_ptr.read().split('\n')[:-1]
        file_ptr.close()
        self.actions_dict = dict()
        for a in actions:
            self.actions_dict[a.split()[1]] = int(a.split()[0])

        super().__init__(file_path, pipeline, **kwargs)

    def parse_file_paths(self, input_path):
        file_ptr = open(self.file_path, 'r')
        info = file_ptr.read().split('\n')[:-1]
        file_ptr.close()
        return info

    def load_file(self):
        """Load index file to get video information."""
        video_names = self.parse_file_paths(self.file_path)
        info = []
        for video_name_fix in video_names:
            video_name = video_name_fix.split('.')[0]
            label_path = os.path.join(self.gt_path, video_name + '.txt')

            video_path = os.path.join(self.videos_path, video_name + '.mp4')
            if not osp.isfile(video_path):
                video_path = os.path.join(self.videos_path, video_name + '.avi')
                if not osp.isfile(video_path):
                    raise NotImplementedError
            file_ptr = open(label_path, 'r')
            content = file_ptr.read().split('\n')[:-1]
            classes = np.zeros(len(content), dtype='int64')
            for i in range(len(content)):
                classes[i] = self.actions_dict[content[i]]
            info.append(
                dict(filename=video_path, labels=classes,
                     video_name=video_name))
        return info

    # def prepare_train(self, idx):
    #     """TRAIN & VALID. Prepare the data for training/valid given the index."""
    #     #Try to catch Exception caused by reading corrupted video file
    #     for ir in range(self.num_retries):
    #         try:
    #             results = copy.deepcopy(self.info[idx])
    #             results = self.pipeline(results)
    #         except Exception as e:
    #             #logger.info(e)
    #             if ir < self.num_retries - 1:
    #                 logger.info(
    #                     "Error when loading {}, have {} trys, will try again".
    #                     format(results['filename'], ir))
    #             idx = random.randint(0, len(self.info) - 1)
    #             continue
    #         return results['imgs'], results['labels'], results['video_name']

    def prepare_train(self, idx):
        """TRAIN & VALID. Prepare the data for training/valid given the index."""
        #Try to catch Exception caused by reading corrupted video file
        for ir in range(self.num_retries):
            # try:
            results = copy.deepcopy(self.info[idx])
            # stream input
            videolen = len(results['labels'])

            batch_res_img = []
            for start_frame in range(0, videolen, self.sample_len):
                results['start_frame'] = start_frame
                results['end_frame'] = start_frame + self.sample_len

                results = self.pipeline(results)

                batch_res_img.append(
                    np.expand_dims(results['imgs'], axis=0).copy())

            stream_imgs = np.concatenate(batch_res_img, axis=0).copy()
            results['imgs'] = copy.deepcopy(stream_imgs)

            # except Exception as e:
            #     #logger.info(e)
            #     if ir < self.num_retries - 1:
            #         logger.info(
            #             "Error when loading {}, have {} trys, will try again".
            #             format(results['filename'], ir))
            #     idx = random.randint(0, len(self.info) - 1)
            #     continue
            return results['imgs'], results['labels'], results['video_name']

    def prepare_test(self, idx):
        for ir in range(self.num_retries):
            try:
                results = copy.deepcopy(self.info[idx])
                # stream input
                videolen = len(results['labels'])

                batch_res_img = []
                for start_frame in range(0, videolen, self.sample_len):
                    results['start_frame'] = start_frame
                    results['end_frame'] = start_frame + self.sample_len

                    results = self.pipeline(results)

                    batch_res_img.append(
                        np.expand_dims(results['imgs'], axis=0).copy())

                stream_imgs = np.concatenate(batch_res_img, axis=0).copy()
                results['imgs'] = copy.deepcopy(stream_imgs)

            except Exception as e:
                #logger.info(e)
                if ir < self.num_retries - 1:
                    logger.info(
                        "Error when loading {}, have {} trys, will try again".
                        format(results['filename'], ir))
                idx = random.randint(0, len(self.info) - 1)
                continue
        return results['imgs'], results['labels'], results['video_name']
