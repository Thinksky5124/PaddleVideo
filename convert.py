import torch
import paddle
import numpy as np

b = paddle.load("data/ctrgcn_from_paddle_ntucs.pdparams")


def transfer():
    input_fp = "data/ctrgcn.pt"
    output_fp = "data/ctrgcn_from_paddle_ntucs.pdparams"
    input_pd = paddle.load("data/CTRGCN_ntucs.pdparams")
    torch_dict = torch.load(input_fp)
    paddle_dict = {}
    for pd_key in input_pd:
        # rename key
        rename_key = pd_key.split('.')[1:]
        key_rename = ''
        for prefix in rename_key:
            key_rename = key_rename + prefix + '.'
        rename_key = key_rename[:-1]
        if "_variance" in rename_key:
            key_prefix = rename_key.split('.')[:-1]
            key_rename = ''
            for prefix in key_prefix:
                key_rename = key_rename + prefix + '.'
            rename_key = key_rename + 'running_var'
        if "_mean" in rename_key:
            key_prefix = rename_key.split('.')[:-1]
            key_rename = ''
            for prefix in key_prefix:
                key_rename = key_rename + prefix + '.'
            rename_key = key_rename + 'running_mean'
        weight = torch_dict[rename_key].cpu().detach().numpy()
        if "fc.weight" in pd_key:
            print("weight {} need to be trans".format(rename_key))
            weight = weight.transpose()
        paddle_dict[pd_key] = weight
    paddle.save(paddle_dict, output_fp)


if __name__ == '__main__':
    transfer()
