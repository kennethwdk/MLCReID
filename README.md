

# MLCReID

The implementation for the *Unsupervised Person Re-identification via Multi-label Classification*, which is accepted by CVPR2020

## Preparetion

### Prerequisites

* Python 3.7
* Pytorch 1.3
* Torchvision 0.5
* Easydict 1.9
* Yaml 0.1

### Data preparation

please refer to [ECN](https://github.com/zhunzhong07/ECN) to prepare dataset, the file structure is like

```
MLCReID/data
├── market
│   └── boundingbox_train
│   └── boundingbox_test
│   └── bounding_box_train_camstyle
│   └── query
├── duke
│   └── boundingbox_train
│   └── boundingbox_test
│   └── bounding_box_train_camstyle
│   └── query
└── msmt17
│   └── boundingbox_train
│   └── boundingbox_test
│   └── bounding_box_train_camstyle
│   └── query
```

### Pretrained model

ResNet-50 pretrained on ImageNet is needed for model initialization, download it and put it into *models/imagenet* dictionary (you can aslo omit this step by changing the code in *resnet.py* so that torchvision will automatically download it)

```shell
mkdir models
mkdir models/imagenet
```

The file tree should be

```
MLCReID/models
└── imagenet
    └── resnet50-19c8e357.pth
```

### Logs and Output

The training logs and checkpoints are saved in *output* dictionary.

```shell
mkdir output
```

## Train and Test

We utilize 1 GTX-2080TI GPU for model training, the hyper-parameters are set in configure files in *experiments* dictionary.

For example, training on Market-1501:

```shell
python tools/train.py --experiments experiments/market.yml --gpus 0
```

If you want to train model on DukeMTMC-reID or MSMT17, just replace the configure files.
```shell
python tools/train.py --experiments experiments/duke.yml --gpus 0
```
```shell
python tools/train.py --experiments experiments/msmt17.yml --gpus 0
```

If you want to test model, change the MODEL_FILE in configure file to the model path, then run following command:

```shell
python tools/test.py --experiments experiments/market.yml --gpus 0
```
You can also find these commands in *scripts* dictionary.

## References

* The code is mainly encouraged by [ECN](https://github.com/zhunzhong07/ECN) and [HRNet](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch) 

