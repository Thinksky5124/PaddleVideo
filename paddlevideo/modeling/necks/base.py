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

import numpy as np
from abc import abstractmethod

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddlevideo.utils import get_logger, get_dist_info

logger = get_logger("paddlevideo")


class BaseNeck(nn.Layer):
    """Base class for neck part.

    All neck should subclass it.
    All subclass should overwrite:

    - Methods: ```init_weights```, initializing weights.
    - Methods: ```forward```, forward function.

    Args:

    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x):
        """Define how the head is going to run.
        """
        raise NotImplemented
