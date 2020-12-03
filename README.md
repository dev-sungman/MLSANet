# Multi-Label Siamese Attention Network for Chest X-ray





## Directory Architecture

**Root**

|---------- models

|---------- 4class_datasets_real_512_test.json (you have to copy from NAS)

|---------- 4class_datasets_real_512_train.json (you have to copy from NAS)

|---------- config.py

|---------- datasets.py

|---------- README.md

|---------- test.py

|---------- train.py

|---------- utils.py

|---------- runs (if you run the train code, it will be made automatically)

|---------- checkpoints (if you run the train code, it will be made automatically)



## Prepare Datasets

* Dataset : /nas125/cleansing_datasets/baseline_followup_pair_4class
* Please move *.json file into your root directory.



## Train

Before you training, please prepare datasets.

* --msg : for log message
* --print_freq : print frequency

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --msg=adam_change+disease+orth --batch_size=20 --print_freq=300
```



* If you want to resume training, please follow below codes.

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --msg=sgd_change+disease+orth --resume=True --batch_size=20 --print_freq=300 --pretrained=checkpoints/2020-09-29_031838_sgd_change+disease+0.5orth_gradclip_real/10153.pt
```



## Test

```bash
CUDA_VISIBLE_DEVICES=0 python test.py --msg=test --batch_size=6 --pretrained checkpoints/2020-10-13_113615_sgd_change+disease+orth_res152_real/20306.pth
```



## Visualize

If you trained model, you can find the tensorboard file in runs/*
