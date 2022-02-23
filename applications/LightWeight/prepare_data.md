# Split Train
## prepare data
```bash
# gtea
python applications/LightWeight/transform_segmentation_label.py data/gtea data/gtea/groundTruth data/gtea --mode localization
python applications/LightWeight/prepare_video_recognition_data.py data/gtea/label.json data/gtea/Videos data/gtea --negative_sample_num 60
# ! have background class

# 50salads
python applications/LightWeight/transform_segmentation_label.py data/50salads data/50salads/groundTruth data/50salads --mode localization
python applications/LightWeight/prepare_video_recognition_data.py data/50salads/label.json data/50salads/Videos data/50salads --negative_sample_num 0 --sample_rate 0.3
# ! no background class

# breakfast

```

GTEA:

mean RGB :[0.5505552534004328, 0.42423616561376576, 0.17930791124574694]

std RGB : [0.13311456349527262, 0.14092562889239943, 0.12356268405634434]

50salads:

mean RGB ∶[0.5139909998345553, 0.5117725498677757，0.4798814301515671]
std RGB :[0.23608918491478523, 0.23385714300069754, 0.23755006337414028]

## train model
```bash
# gtea
# single gpu
export CUDA_VISIBLE_DEVICES=2
python main.py  --validate -c applications/LightWeight/config/split/gtea/tsm_gtea.yaml --seed 0
# multi gpu
export CUDA_VISIBLE_DEVICES=2,3
python -B -m paddle.distributed.launch --gpus="2,3"  --log_dir=./output main.py  --validate -c applications/LightWeight/config/split/gtea/tsm_gtea.yaml

# 50salads
# single gpu
export CUDA_VISIBLE_DEVICES=2
python main.py  --validate -c applications/LightWeight/config/split/50salads/tsm_50salads.yaml --seed 0
# multi gpu
export CUDA_VISIBLE_DEVICES=2,3
python -B -m paddle.distributed.launch --gpus="2,3"  --log_dir=./output main.py  --validate -c applications/LightWeight/config/split/50salads/tsm_50salads.yaml
```

## extracte feture
```bash
# export infer model
python tools/export_model.py -c applications/LightWeight/config/split/50salads/tsm_extractor_50salads.yaml \
                                -p output/TSM/TSM_best.pdparams \
                                -o inference/TSM

# use infer model to extract video feature
python applications/LightWeight/extractor.py --input_file data/50salads/Videos \
                           --output_path data/50salads/extract_features \
                           --gt_path data/50salads/groundTruth \
                           --config applications/LightWeight/config/split/50salads/tsm_extractor_50salads.yaml \
                           --model_file inference/TSM/TSM.pdmodel \
                           --params_file inference/TSM/TSM.pdiparams \
                           --use_gpu=True \
                           --use_tensorrt=False \
                           --batch_size=8
```
## segmentation model train
```bash
# gtea
export CUDA_VISIBLE_DEVICES=2
python main.py  --validate -c applications/LightWeight/config/split/gtea/ms_tcn_GTEA.yaml --seed 0
python main.py  --validate -c applications/LightWeight/config/split/gtea/asrf_GTEA.yaml --seed 0

# 50salads
python main.py  --validate -c applications/LightWeight/config/split/50salads/ms_tcn_50salads.yaml --seed 0
```

# one-shot train

## prepare data
```bash
# gtea
python applications/LightWeight/prepare_ete_data_list.py \
                        --split_list_path data/gtea/splits \
                        --label_path data/gtea/groundTruth \
                        --output_path data/gtea/split_frames \
                        --window_size 60 \
                        --strike 15
```


## train model
```bash
# gtea
# single gpu
export CUDA_VISIBLE_DEVICES=2
python main.py  --validate -c applications/LightWeight/config/one_shot/gtea/ete_tsm_mstcn.yaml --seed 0
# multi gpu
export CUDA_VISIBLE_DEVICES=2,3
python -B -m paddle.distributed.launch --gpus="2,3"  --log_dir=./output main.py  --validate -c applications/LightWeight/config/one_shot/gtea/ete_tsm_mstcn.yaml --seed 0

# 50salads

```
## test model
```bash
python main.py  --test -c applications/LightWeight/config/one_shot/gtea/ete_tsm_mstcn.yaml --weights=./output/ETEMSTCN/ETEMSTCN_best.pdparams
```
