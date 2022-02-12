import argparse
import os
from os import path as osp
from paddle import inference
from paddle.inference import Config, create_predictor

import cv2
import decord as de
import imageio
import matplotlib as mpl
import matplotlib.cm as cm
import numpy as np
import paddle
import paddle.nn.functional as F
import pandas
from PIL import Image
import sys
from tqdm import tqdm

sys.path.append("./")
from paddlevideo.utils import get_config
from paddlevideo.loader.pipelines import (
    AutoPadding, CenterCrop, DecodeSampler, FeatureDecoder, FrameDecoder,
    GroupResize, Image2Array, ImageDecoder, JitterScale, MultiCrop,
    Normalization, PackOutput, Scale, VideoDecoder, Sampler, TenCrop, ToArray,
    UniformCrop, SegmentationSampler)


def parse_args():

    def str2bool(v):
        return v.lower() in ("true", "t", "1")

    # general params
    parser = argparse.ArgumentParser("PaddleVideo Inference model script")
    parser.add_argument('-c',
                        '--config',
                        type=str,
                        default='configs/example.yaml',
                        help='config file path')
    parser.add_argument("-i", "--input_file", type=str, help="input file path")
    parser.add_argument("-o",
                        "--output_path",
                        type=str,
                        help="output file path")
    parser.add_argument("--model_file", type=str)
    parser.add_argument("--params_file", type=str)

    # params for predict
    parser.add_argument("-b", "--batch_size", type=int, default=1)
    parser.add_argument("--use_gpu", type=str2bool, default=True)
    parser.add_argument("--precision", type=str, default="fp32")
    parser.add_argument("--ir_optim", type=str2bool, default=True)
    parser.add_argument("--use_tensorrt", type=str2bool, default=False)
    parser.add_argument("--gpu_mem", type=int, default=8000)
    parser.add_argument("--enable_benchmark", type=str2bool, default=False)
    parser.add_argument("--enable_mkldnn", type=str2bool, default=False)
    parser.add_argument("--cpu_threads", type=int, default=None)
    # parser.add_argument("--hubserving", type=str2bool, default=False)  #TODO

    return parser.parse_args()


def create_paddle_predictor(args, cfg):
    config = Config(args.model_file, args.params_file)
    if args.use_gpu:
        config.enable_use_gpu(args.gpu_mem, 0)
    else:
        config.disable_gpu()
        if args.cpu_threads:
            config.set_cpu_math_library_num_threads(args.cpu_threads)
        if args.enable_mkldnn:
            # cache 10 different shapes for mkldnn to avoid memory leak
            config.set_mkldnn_cache_capacity(10)
            config.enable_mkldnn()
            if args.precision == "fp16":
                config.enable_mkldnn_bfloat16()

    # config.disable_glog_info()
    config.switch_ir_optim(args.ir_optim)  # default true
    if args.use_tensorrt:
        # choose precision
        if args.precision == "fp16":
            precision = inference.PrecisionType.Half
        elif args.precision == "int8":
            precision = inference.PrecisionType.Int8
        else:
            precision = inference.PrecisionType.Float32

        # calculate real max batch size during inference when tenrotRT enabled
        max_batch_size = args.batch_size
        if 'num_seg' in cfg.INFERENCE:
            # num_seg: number of segments when extracting frames.
            # seg_len: number of frames extracted within a segment, default to 1.
            # num_views: the number of video frame groups obtained by cropping and flipping,
            # uniformcrop=3, tencrop=10, centercrop=1.
            num_seg = cfg.INFERENCE.num_seg
            seg_len = cfg.INFERENCE.get('seg_len', 1)
            num_views = 1
            if 'tsm' in cfg.model_name.lower():
                num_views = 1  # CenterCrop
            elif 'tsn' in cfg.model_name.lower():
                num_views = 10  # TenCrop
            elif 'timesformer' in cfg.model_name.lower():
                num_views = 3  # UniformCrop
            elif 'videoswin' in cfg.model_name.lower():
                num_views = 3  # UniformCrop
            max_batch_size = args.batch_size * num_views * num_seg * seg_len
        config.enable_tensorrt_engine(precision_mode=precision,
                                      max_batch_size=max_batch_size)

    config.enable_memory_optim()
    # use zero copy
    config.switch_use_feed_fetch_ops(False)

    predictor = create_predictor(config)

    return config, predictor


class FeatureExtractorSampler(Sampler):
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
                 num_seg,
                 seg_len,
                 frame_interval=None,
                 valid_mode=False,
                 select_left=False,
                 dense_sample=False,
                 linspace_sample=False,
                 use_pil=True):
        super(FeatureExtractorSampler,
              self).__init__(num_seg,
                             seg_len,
                             frame_interval=frame_interval,
                             valid_mode=valid_mode,
                             select_left=select_left,
                             dense_sample=dense_sample,
                             linspace_sample=linspace_sample,
                             use_pil=use_pil)

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

        if end_frame > frames_len:
            end_frame = frames_len

        if results['format'] == 'video':
            frames_idx = list(range(start_frame, end_frame))

        else:
            raise NotImplementedError

        return self._get(frames_idx, results)


class Inference_helper():

    def __init__(self,
                 num_seg=8,
                 seg_len=1,
                 short_size=256,
                 target_size=224,
                 top_k=1,
                 backend='decord'):
        self.num_seg = num_seg
        self.seg_len = seg_len
        self.short_size = short_size
        self.target_size = target_size
        self.top_k = top_k
        self.backend = backend

    def postprocess(self, output, output_path, feature_dim, print_output=True):
        """
        output: list
        """
        output_np = np.concatenate(
            output[:-1],
            axis=0)  # [clip_num, batch_size * num_segs, in_channels]
        feature_dim_T = np.reshape(output_np, (-1, feature_dim))
        last_batch_feature_T = output[-1][0].squeeze()
        last_batch_feature_T = last_batch_feature_T[:(
            last_batch_feature_T.shape[0] - self.offsets), :]
        feature_dim_T = np.concatenate([feature_dim_T, last_batch_feature_T],
                                       axis=0)
        feature_dim = feature_dim_T.T

        save_path = os.path.join(output_path, self.input_file + '.npy')
        # output shape [2048, T] npy
        np.save(save_path, feature_dim)

    def preprocess(self, input_file, start_frame_list, end_frame_list, img_mean,
                   img_std):
        """
        input_file: str, file path
        return: list
        """
        assert os.path.isfile(input_file) is not None, "{0} not exists".format(
            input_file)
        self.input_file = input_file.split('/')[-1].split('.')[0]

        ops = [
            VideoDecoder(backend=self.backend),
            FeatureExtractorSampler(self.num_seg, self.seg_len,
                                    valid_mode=True),
            Scale(self.short_size),
            CenterCrop(self.target_size),
            Image2Array(),
            Normalization(img_mean, img_std)
        ]

        batch_res = []
        for start_frame, end_frame in zip(start_frame_list, end_frame_list):
            results = {
                'filename': input_file,
                'start_frame': start_frame,
                'end_frame': end_frame
            }
            for op in ops:
                results = op(results)
            imgs_shape = list(results['imgs'].shape)
            if imgs_shape[0] < self.num_seg:
                offsets = self.num_seg - results['imgs'].shape[0]
                self.offsets = offsets
                imgs_shape[0] = offsets
                offset_img = np.zeros(imgs_shape, dtype=results['imgs'].dtype)
                imgs = np.concatenate([results['imgs'], offset_img], axis=0)
                results['imgs'] = imgs.copy()
            batch_res.append(np.expand_dims(results['imgs'], axis=0).copy())
        res = np.concatenate(batch_res, axis=0).copy()
        return [res]


def parse_file_paths(input_path: str) -> list:
    if osp.isfile(input_path):
        files = [
            input_path,
        ]
    else:
        files = os.listdir(input_path)
        files = [
            file for file in files
            if (file.endswith(".avi") or file.endswith(".mp4"))
        ]
        files = [osp.join(input_path, file) for file in files]
    return files


def main():
    args = parse_args()
    cfg = get_config(args.config, show=False)

    model_name = cfg.model_name
    print(f"Inference model({model_name})...")
    InferenceHelper = Inference_helper(num_seg=cfg.INFERENCE.num_seg,
                                       target_size=cfg.INFERENCE.target_size)

    _, predictor = create_paddle_predictor(args, cfg)

    # get input_tensor and output_tensor
    input_names = predictor.get_input_names()
    output_names = predictor.get_output_names()
    input_tensor_list = []
    output_tensor_list = []
    for item in input_names:
        input_tensor_list.append(predictor.get_input_handle(item))
    for item in output_names:
        output_tensor_list.append(predictor.get_output_handle(item))

    files = parse_file_paths(args.input_file)

    img_mean = cfg.INFERENCE.mean
    img_std = cfg.INFERENCE.std

    # Inferencing process
    batch_num = args.batch_size
    for file_path in tqdm(files, desc='feature extract:'):

        cap = cv2.VideoCapture(file_path)
        videolen = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        video_outputs = []
        for batch_start_frame in tqdm(
                range(0, videolen, cfg.INFERENCE.num_seg * batch_num),
                desc=file_path.split('/')[-1].split('.')[0]):
            start_frame_list = []
            end_frame_list = []
            start_frame = batch_start_frame
            for batch_id in range(batch_num):
                if start_frame <= videolen:
                    end_frame = start_frame + cfg.INFERENCE.num_seg
                    start_frame_list.append(start_frame)
                    end_frame_list.append(end_frame)
                    start_frame = end_frame
                else:
                    break
            # Pre process batched input
            batched_inputs = InferenceHelper.preprocess(file_path,
                                                        start_frame_list,
                                                        end_frame_list,
                                                        img_mean, img_std)

            # run inference
            for i in range(len(input_tensor_list)):
                input_tensor_list[i].copy_from_cpu(batched_inputs[i])
            predictor.run()

            batched_outputs = []
            for j in range(len(output_tensor_list)):
                batched_outputs.append(output_tensor_list[j].copy_to_cpu())

            video_outputs.append(batched_outputs.copy())

        InferenceHelper.postprocess(
            video_outputs,
            args.output_path,
            cfg.INFERENCE.feature_dim,
            not args.enable_benchmark,
        )

        # time.sleep(0.01)  # sleep for T4 GPU


if __name__ == "__main__":
    main()
