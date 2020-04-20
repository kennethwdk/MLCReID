import os
import yaml

from easydict import EasyDict as edict

config = edict()

config.CONFIG_NAME = ''
config.OUTPUT_DIR = ''
config.DATA_DIR = ''
config.GPUS = '0'
config.CUDA = True
config.WORKERS = 8
config.PRINT_FREQ = 20

# Cudnn related params
config.CUDNN = edict()
config.CUDNN.BENCHMARK = True
config.CUDNN.DETERMINISTIC = False
config.CUDNN.ENABLED = True

# common params for NETWORK
config.MODEL = edict()
config.MODEL.NAME = 'resnet50'
config.MODEL.INIT_WEIGHTS = True
config.MODEL.PRETRAINED = ''
config.MODEL.IMAGE_SIZE = [256, 128]  
config.MODEL.FEATURES = 2048
config.MODEL.CLASSES = 751

config.MMCL = edict()
config.MMCL.DELTA = 5
config.MMCL.R = 0.01

config.MPLP = edict()
config.MPLP.T = 0.6

# DATASET related params
config.DATASET = edict()
config.DATASET.ROOT = ''
config.DATASET.DATASET = 'market'
config.DATASET.DATA_FORMAT = 'jpg'
# training data augmentation
config.DATASET.RE = 0.5

# train
config.TRAIN = edict()
config.TRAIN.LR = 0.1
config.TRAIN.LR_STEP = 40
config.TRAIN.LR_FACTOR = 0.1

config.TRAIN.OPTIMIZER = 'sgd'
config.TRAIN.MOMENTUM = 0.9
config.TRAIN.WEIGHT_DECAY = 0.0005
config.TRAIN.NESTEROV = True

config.TRAIN.BEGIN_EPOCH = 0
config.TRAIN.END_EPOCH = 60

config.TRAIN.RESUME = False
config.TRAIN.CHECKPOINT = ''

config.TRAIN.BATCH_SIZE = 128
config.TRAIN.SHUFFLE = True

# testing
config.TEST = edict()
config.TEST.BATCH_SIZE = 32
config.TEST.MODEL_FILE = ''
config.TEST.OUTPUT_FEATURES = 'pool5'

def _update_dict(k, v):
    for vk, vv in v.items():
        if vk in config[k]:
            config[k][vk] = vv
        else:
            raise ValueError("{}.{} not exist in config.py".format(k, vk))

def update_config(config_file):
    exp_config = None
    with open(config_file) as f:
        exp_config = edict(yaml.load(f))
        for k, v in exp_config.items():
            if k in config:
                if isinstance(v, dict):
                    _update_dict(k, v)
                else:
                    if k == 'SCALES':
                        config[k][0] = (tuple(v))
                    else:
                        config[k] = v
            else:
                raise ValueError("{} not exist in config.py".format(k))

def gen_config(config_file):
    cfg = dict(config)
    for k, v in cfg.items():
        if isinstance(v, edict):
            cfg[k] = dict(v)

    with open(config_file, 'w') as f:
        yaml.dump(dict(cfg), f, default_flow_style=False)

if __name__ == '__main__':
    import sys
    gen_config(sys.argv[1])