import argparse
import os
import pandas as pd

from tqdm import tqdm

background_class = ['None', 'background']

def generate_clip_dict(vid, gt_content, video_len, window_size, strike, end_overlap=True, mode='slide_windows'):
    if mode == 'slide_windows':
        return generate_slide_windows_clip_dict(vid, video_len, window_size, strike, end_overlap=end_overlap)
    elif mode == 'action_clip':
        return generate_action_clip_dict(vid, gt_content)
    else:
        raise NotImplementedError

def resample_background(video_clip_dict, neg_num):
    bg_dict = {'bg_vid': video_clip_dict['bg_vid'], 'bg_start_frame': video_clip_dict['bg_start_frame'], 'bg_end_frame':video_clip_dict['bg_end_frame']}

    clip = pd.DataFrame(bg_dict)
    clip = clip.sample(frac=1)
    clips_dict_from_pd = clip.to_dict()
    clips_dict = {}
    vid_list = list(clips_dict_from_pd['bg_vid'].values())[:neg_num]
    start_frame_list = list(clips_dict_from_pd['bg_start_frame'].values())[:neg_num]
    end_frame_list = list(clips_dict_from_pd['bg_end_frame'].values())[:neg_num]

    clips_dict['vid'] = video_clip_dict['vid'] + vid_list
    clips_dict['start_frame'] = video_clip_dict['start_frame'] + start_frame_list
    clips_dict['end_frame'] = video_clip_dict['end_frame'] + end_frame_list
    return clips_dict

def get_labels_start_end_time(frame_wise_labels, bg_class=background_class):
    labels = []
    starts = []
    ends = []
    bg_starts = []
    bg_ends = []

    last_label = frame_wise_labels[0]
    if frame_wise_labels[0] not in bg_class:
        labels.append(frame_wise_labels[0])
        starts.append(0)
    else:
        bg_starts.append(0)
    for i in range(len(frame_wise_labels)):
        if frame_wise_labels[i] != last_label:
            if frame_wise_labels[i] not in bg_class:
                labels.append(frame_wise_labels[i])
                starts.append(i)
            else:
                bg_starts.append(i)
            if last_label not in bg_class:
                ends.append(i)
            else:
                bg_ends.append(i)
            last_label = frame_wise_labels[i]
    if last_label not in bg_class:
        ends.append(i + 1)
    else:
        bg_ends.append(i + 1)
    return labels, starts, ends, bg_starts, bg_ends

def generate_action_clip_dict(vid, gt_content):
    clip_dict = {}

    _, starts, ends, bg_starts, bg_ends = get_labels_start_end_time(gt_content)

    clip_dict['vid'] = [vid] * len(starts)
    clip_dict['start_frame'] = starts
    clip_dict['end_frame'] = ends
    
    clip_dict['bg_vid'] = [vid] * len(bg_starts)
    clip_dict['bg_start_frame'] = bg_starts
    clip_dict['bg_end_frame'] = bg_ends
    return clip_dict

def generate_slide_windows_clip_dict(vid, video_len, window_size, strike, end_overlap=True):
    clip_dict = {}
    vid_list = []
    start_frame_list = []
    end_frame_list = []
    for start_frame in range(0, video_len, strike):
        if start_frame + window_size > video_len:
            break

        end_frame = start_frame + window_size
        vid_list.append(vid)
        start_frame_list.append(start_frame)
        end_frame_list.append(end_frame)

    if end_overlap:
        end_frame = video_len
        start_frame = video_len - window_size
        vid_list.append(vid)
        start_frame_list.append(start_frame)
        end_frame_list.append(end_frame)
    else:
        end_frame = video_len
        vid_list.append(vid)
        start_frame_list.append(start_frame)
        end_frame_list.append(end_frame)
    clip_dict['vid'] = vid_list
    clip_dict['start_frame'] = start_frame_list
    clip_dict['end_frame'] = end_frame_list
    return clip_dict


def get_arguments():
    """
    parse all the arguments from command line inteface
    return a list of parsed arguments
    """

    parser = argparse.ArgumentParser(
        description="convert segmentation and localization label")
    parser.add_argument("--split_list_path",
                        type=str,
                        help="path of a split train test file")
    parser.add_argument("--label_path", type=str, help="path of a label file")
    parser.add_argument(
        "--output_path",
        type=str,
        help="output path of split_list.",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        help="path of output file.",
    )
    parser.add_argument(
        "--strike",
        type=int,
        default=1,
        help="slide windows strike size.",
    )
    parser.add_argument(
        "--end_overlap",
        type=bool,
        default=True,
        help="end of video can overlap.",
    )
    parser.add_argument(
        "--split_mode",
        type=str,
        default='slide_windows',
        help="end of video can overlap.",
    )
    parser.add_argument(
        "--neg_num",
        type=int,
        default=60,
        help="end of video can overlap.",
    )

    return parser.parse_args()


def main():
    args = get_arguments()

    path = os.path.join(args.output_path)
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        print(path + ' 创建成功')

    files = os.listdir(args.split_list_path)
    train_files = [
        file for file in files
        if (file.endswith(".bundle") and file.startswith('train'))
    ]
    train_files = [
        os.path.join(args.split_list_path, file) for file in train_files
    ]

    test_files = [
        file for file in files
        if (file.endswith(".bundle") and file.startswith('test'))
    ]
    test_files = [
        os.path.join(args.split_list_path, file) for file in test_files
    ]

    for train_file in tqdm(train_files, desc='train dataset:'):
        file_ptr = open(train_file, 'r')
        video_name_list = file_ptr.read().split('\n')[:-1]
        file_ptr.close()

        train_file_name = train_file.split('/')[-1]
        # output path
        clip_list_path = os.path.join(args.output_path, train_file_name)
        total_clip_dict = {}
        for video_name in video_name_list:
            # read gt file
            file_name = video_name.split('.')[0] + ".txt"
            gt_file_path = os.path.join(args.label_path, file_name)
            file_ptr = open(gt_file_path, 'r')
            content = file_ptr.read().split('\n')[:-1]
            file_ptr.close()
            video_len = len(content)
            # split video to clip dcit
            clip_dict = generate_clip_dict(video_name,
                                           content,
                                           video_len,
                                           window_size=args.window_size,
                                           strike=args.strike,
                                           end_overlap=args.end_overlap,
                                           mode=args.split_mode)
            for name, value in clip_dict.items():
                if name not in total_clip_dict.keys():
                    total_clip_dict[name] = []
                else:
                    total_clip_dict[
                        name] = total_clip_dict[name] + clip_dict[name]

        if args.split_mode == 'action_clip':
            total_clip_dict = resample_background(total_clip_dict, neg_num=args.neg_num)

        recog_content = []
        for vid, start_frame, end_frame in zip(total_clip_dict['vid'],
                                               total_clip_dict['start_frame'],
                                               total_clip_dict['end_frame']):
            recog_content.append(vid + ' ' + str(start_frame) + ' ' +
                                 str(end_frame) + '\n')
        f = open(clip_list_path, "w")
        f.writelines(recog_content)
        f.close()

    for test_file in tqdm(test_files, desc='test dataset:'):
        file_ptr = open(test_file, 'r')
        video_name_list = file_ptr.read().split('\n')[:-1]
        file_ptr.close()

        test_file_name = test_file.split('/')[-1]
        # output path
        clip_list_path = os.path.join(args.output_path, test_file_name)
        total_clip_dict = {}
        for video_name in video_name_list:
            # read gt file
            file_name = video_name.split('.')[0] + ".txt"
            gt_file_path = os.path.join(args.label_path, file_name)
            file_ptr = open(gt_file_path, 'r')
            content = file_ptr.read().split('\n')[:-1]
            file_ptr.close()
            video_len = len(content)
            # split video to clip dict
            clip_dict = generate_clip_dict(video_name,
                                           content,
                                           video_len,
                                           window_size=args.window_size,
                                           strike=args.window_size,
                                           end_overlap=False,
                                           mode='slide_windows')
            for name, value in clip_dict.items():
                if name not in total_clip_dict.keys():
                    total_clip_dict[name] = []
                else:
                    total_clip_dict[
                        name] = total_clip_dict[name] + clip_dict[name]

        recog_content = []
        for vid, start_frame, end_frame in zip(total_clip_dict['vid'],
                                               total_clip_dict['start_frame'],
                                               total_clip_dict['end_frame']):
            recog_content.append(vid + ' ' + str(start_frame) + ' ' +
                                 str(end_frame) + '\n')
        f = open(clip_list_path, "w")
        f.writelines(recog_content)
        f.close()


if __name__ == "__main__":
    main()
