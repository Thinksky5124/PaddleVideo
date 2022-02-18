import argparse
import os

from tqdm import tqdm


def generate_clip_list(vid, video_len, window_size, strike, end_overlap=True):
    clip_list = []
    for start_frame in range(0, video_len, strike):
        if start_frame + window_size > video_len:
            break

        end_frame = start_frame + window_size
        info = vid + ' ' + str(start_frame) + ' ' + str(end_frame)
        clip_list.append(info)

    if end_overlap:
        end_frame = video_len
        start_frame = video_len - window_size
        info = vid + ' ' + str(start_frame) + ' ' + str(end_frame)
        clip_list.append(info)
    else:
        end_frame = video_len
        info = vid + ' ' + str(start_frame) + ' ' + str(end_frame)
        clip_list.append(info)
    return clip_list


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
        if (file.endswith(".bundle") or file.startswith('train'))
    ]
    train_files = [
        os.path.join(args.split_list_path, file) for file in train_files
    ]

    test_files = [
        file for file in files
        if (file.endswith(".bundle") or file.startswith('test'))
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
        total_clip_list = []
        for video_name in video_name_list:
            # read gt file
            file_name = video_name.split('.')[0] + ".txt"
            gt_file_path = os.path.join(args.label_path, file_name)
            file_ptr = open(gt_file_path, 'r')
            content = file_ptr.read().split('\n')[:-1]
            file_ptr.close()
            video_len = len(content)
            # split video to clip list
            clip_list = generate_clip_list(video_name,
                                           video_len,
                                           window_size=args.window_size,
                                           strike=args.strike,
                                           end_overlap=args.end_overlap)
            total_clip_list = total_clip_list + clip_list

        recog_content = [line + "\n" for line in total_clip_list]
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
        total_clip_list = []
        for video_name in video_name_list:
            # read gt file
            file_name = video_name.split('.')[0] + ".txt"
            gt_file_path = os.path.join(args.label_path, file_name)
            file_ptr = open(gt_file_path, 'r')
            content = file_ptr.read().split('\n')[:-1]
            file_ptr.close()
            video_len = len(content)
            # split video to clip list
            clip_list = generate_clip_list(video_name,
                                           video_len,
                                           window_size=args.window_size,
                                           strike=args.strike,
                                           end_overlap=args.end_overlap)
            total_clip_list = total_clip_list + clip_list

        recog_content = [line + "\n" for line in total_clip_list]
        f = open(clip_list_path, "w")
        f.writelines(recog_content)
        f.close()


if __name__ == "__main__":
    main()
