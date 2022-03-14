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
from .base import BaseNeck
from ..registry import NECKS
import numpy as np

from paddle import ParamAttr
from ..weight_init import weight_init_

from ..backbones.ms_tcn import calculate_gain, KaimingUniform_like_torch
from ..backbones.ms_tcn import init_bias, SingleStageModel, DilatedResidualLayer


@NECKS.register()
class ETENeck(BaseNeck):

    def __init__(self,
                 buffer_channels,
                 hidden_channels,
                 num_layers,
                 num_segs=15,
                 clip_buffer_num=3,
                 sliding_strike=15,
                 max_len=10000,
                 data_format="NCHW"):
        super().__init__()

        assert (data_format in [
            'NCHW', 'NHWC'
        ]), f"data_format must be 'NCHW' or 'NHWC', but got {data_format}"

        self.data_format = data_format
        self.buffer_channels = buffer_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.clip_buffer_num = clip_buffer_num
        self.sliding_strike = sliding_strike
        self.max_len = max_len
        self.num_segs = num_segs

        self.avgpool2d = nn.AdaptiveAvgPool2D((1, 1),
                                              data_format=self.data_format)

        self.pos_embedding = PositionalEmbedding(self.buffer_channels,
                                                 self.max_len)

    def forward(self, x, seg_mask, memery_buffer, mask_buffer, start_frame):
        """ ETEHead forward
        """
        # x.shape = [N * num_segs, 2048, 7, 7]
        x = self.avgpool2d(x)
        # x.shape = [N * num_segs, 2048, 1, 1]
        cls_feature = x

        # segmentation branch
        # [N * num_segs, 2048]
        seg_x = paddle.squeeze(x)
        # [N, num_segs, 2048]
        seg_feature = paddle.reshape(seg_x,
                                     shape=[-1, self.num_segs, seg_x.shape[-1]])
        # [N, 2048, num_segs]
        seg_feature = paddle.transpose(seg_feature, perm=[0, 2, 1])

        # # position encoding
        # # [N, num_segs, buffer_channels]
        # end_frame = start_frame + self.num_segs
        # pos_emb = self.pos_embedding(seg_feature.shape[0], start_frame, end_frame)
        # # [N, buffer_channels, num_segs]
        # pos_emb = paddle.transpose(pos_emb, perm=[0, 2, 1])
        # seg_feature = paddle.concat([seg_feature, pos_emb], axis=1)

        # memery model
        # if memery_buffer is None:
        #     # [N, buffer_channels, num_segs * clip_buffer_num]
        #     zeros_pad = paddle.zeros((seg_feature.shape[0], self.buffer_channels, self.num_segs * self.clip_buffer_num))
        #     mask_zeros_pad = paddle.zeros((seg_mask.shape[0], seg_mask.shape[1], self.num_segs * self.clip_buffer_num))
        #     # [N, buffer_channels, num_segs * (clip_buffer_num + 1)]
        #     seg_feature = paddle.concat([zeros_pad, seg_feature], axis=2)
        #     pad_mask = paddle.concat([mask_zeros_pad, seg_mask],axis=2)
        #     # [N, buffer_channels, num_segs * clip_buffer_num]
        #     memery_buffer = paddle.roll(seg_feature, shifts=self.sliding_strike, axis=2)[:, :, :(self.num_segs * self.clip_buffer_num)].clone().detach()
        #     mask_buffer = paddle.roll(pad_mask, shifts=self.sliding_strike, axis=2)[:, :, :(self.num_segs * self.clip_buffer_num)].clone()
        # else:
        #     # [N, buffer_channels, num_segs * (clip_buffer_num + 1)]
        #     seg_feature = paddle.concat([memery_buffer, seg_feature], axis=2)
        #     pad_mask = paddle.concat([mask_buffer, seg_mask],axis=2)
        #     # [N, buffer_channels, num_segs * clip_buffer_num]
        #     memery_buffer = paddle.roll(seg_feature, shifts=self.sliding_strike, axis=2)[:, :, :(self.num_segs * self.clip_buffer_num)].clone().detach()
        #     mask_buffer = paddle.roll(pad_mask, shifts=self.sliding_strike, axis=2)[:, :, :(self.num_segs * self.clip_buffer_num)].clone()
        return seg_feature, cls_feature, memery_buffer, mask_buffer

    def init_weights(self):
        for layer in self.sublayers():
            if isinstance(layer, nn.Conv1D):
                layer.weight.set_value(
                    KaimingUniform_like_torch(layer.weight).astype('float32'))
                if layer.bias is not None:
                    layer.bias.set_value(
                        init_bias(layer.weight, layer.bias).astype('float32'))


def position_encoding_init(n_position, d_pos_vec, dtype="float32"):
    """
    Generates the initial values for the sinusoidal position encoding table.
    This method follows the implementation in tensor2tensor, but is slightly
    different from the description in "Attention Is All You Need".
    Args:
        n_position (int):
            The largest position for sequences, that is, the maximum length
            of source or target sequences.
        d_pos_vec (int):
            The size of positional embedding vector.
        dtype (str, optional):
            The output `numpy.array`'s data type. Defaults to "float32".
    Returns:
        numpy.array:
            The embedding table of sinusoidal position encoding with shape
            `[n_position, d_pos_vec]`.
    Example:
        .. code-block::
            from paddlenlp.transformers import position_encoding_init
            max_length = 256
            emb_dim = 512
            pos_table = position_encoding_init(max_length, emb_dim)
    """
    channels = d_pos_vec
    position = np.arange(n_position)
    num_timescales = channels // 2
    log_timescale_increment = (np.log(float(1e4) / float(1)) /
                               (num_timescales - 1))
    inv_timescales = np.exp(
        np.arange(num_timescales) * -log_timescale_increment)
    scaled_time = np.expand_dims(position, 1) * np.expand_dims(
        inv_timescales, 0)
    signal = np.concatenate([np.sin(scaled_time), np.cos(scaled_time)], axis=1)
    signal = np.pad(signal, [[0, 0], [0, np.mod(channels, 2)]], 'constant')
    position_enc = signal
    return position_enc.astype(dtype)


class PositionalEmbedding(nn.Layer):
    """
    This layer produces sinusoidal positional embeddings of any length.
    While in `forward()` method, this layer lookups embeddings vector of
    ids provided by input `pos`.
    Args:
        emb_dim (int):
            The size of each embedding vector.
        max_length (int):
            The maximum length of sequences.
    """

    def __init__(self, emb_dim, max_length):
        super(PositionalEmbedding, self).__init__()
        self.emb_dim = emb_dim
        self.max_length = max_length

        self.pos_encoder = nn.Embedding(
            num_embeddings=max_length,
            embedding_dim=self.emb_dim,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Assign(
                    position_encoding_init(max_length, self.emb_dim))))

    def forward(self, bs, start_frame, end_frame):
        r"""
        Computes positional embedding.
        Args:
            pos (Tensor):
                The input position ids with shape `[batch_size, sequence_length]` whose
                data type can be int or int64.
        Returns:
            Tensor:
                The positional embedding tensor of shape
                `(batch_size, sequence_length, emb_dim)` whose data type can be
                float32 or float64.

        Example:
            .. code-block::
                import paddle
                from paddlenlp.transformers import PositionalEmbedding
                pos_embedding = PositionalEmbedding(
                    emb_dim=512,
                    max_length=256)
                batch_size = 5
                pos = paddle.tile(paddle.arange(start=0, end=50), repeat_times=[batch_size, 1])
                pos_emb = pos_embedding(pos)
        """
        pos = paddle.tile(paddle.arange(start=0, end=self.max_length),
                          repeat_times=[bs, 1])
        pos_emb = self.pos_encoder(pos)
        pos_emb.stop_gradient = True
        pos_emb = pos_emb[:, start_frame:end_frame, :]
        return pos_emb
