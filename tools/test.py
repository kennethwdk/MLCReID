import argparse
import os
import os.path as osp
import sys
import pprint

import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader

import _init_paths
from lib.datasets.dataset import DataSet
from lib import models
from lib.evaluator import Evaluator
from lib.utils.data import transforms as T
from lib.utils.data.preprocessor import Preprocessor
from lib.utils.logging import Logger
from lib.utils.serialization import load_checkpoint
from lib.utils.config import config, update_config

def parse_args():
    parser = argparse.ArgumentParser(description='Unsupervised ReID via Multi-label Classification')
    parser.add_argument('--experiments', dest='cfg_file',
                        help='optional config file',
                        default='experiments/market.yml', type=str)
    parser.add_argument('--gpus', type=str, help='gpus')
    parser.add_argument('--workers', type=int, help='num of dataloader workers')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    update_config(args.cfg_file)

    if args.gpus:
        config.GPUS = args.gpus
    else:
        config.CUDA = False
    if args.workers:
        config.WORKERS = args.workers
    print('Using config:')
    pprint.pprint(config)

    torch.backends.cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    if config.CUDA:
        os.environ["CUDA_VISIBLE_DEVICES"] = config.GPUS
    device = torch.device('cuda' if config.CUDA else 'cpu')

    # Redirect print to both console and log file
    sys.stdout = Logger(osp.join(config.OUTPUT_DIR, 'log-eval.txt'))

    # Create data loaders
    dataset = DataSet(config.DATASET.ROOT, config.DATASET.DATASET)

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    transformer = T.Compose([
        T.Resize(config.MODEL.IMAGE_SIZE, interpolation=3),
        T.ToTensor(),
        normalizer,
    ])

    query_loader = DataLoader(
        Preprocessor(dataset.query,
                     root=osp.join(dataset.images_dir, dataset.query_path), transform=transformer),
        batch_size=config.TEST.BATCH_SIZE, num_workers=config.WORKERS,
        shuffle=False, pin_memory=True)

    gallery_loader = DataLoader(
        Preprocessor(dataset.gallery,
                     root=osp.join(dataset.images_dir, dataset.gallery_path), transform=transformer),
        batch_size=config.TEST.BATCH_SIZE, num_workers=config.WORKERS,
        shuffle=False, pin_memory=True)

    # Create model
    model = models.create(config.MODEL.NAME)

    # Load from checkpoint
    checkpoint = load_checkpoint(config.TEST.MODEL_FILE)
    print('best model at epoch: {}'.format(checkpoint['epoch']))
    model.load_state_dict(checkpoint['state_dict'], strict=False)

    # Set model
    model = nn.DataParallel(model).to(device)
    
    print('Test with best model:')
    evaluator = Evaluator(model)
    evaluator.evaluate(query_loader, gallery_loader, dataset.query,
                       dataset.gallery, config.TEST.OUTPUT_FEATURES)

if __name__ == '__main__':
    main()
