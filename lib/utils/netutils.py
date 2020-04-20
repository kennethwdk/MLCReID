import torch

def get_optimizer(cfg, model_parameters):
    optimizer = None
    if cfg.TRAIN.OPTIMIZER == 'sgd':
        optimizer = torch.optim.SGD(
            model_parameters,
            lr=cfg.TRAIN.LR,
            momentum=cfg.TRAIN.MOMENTUM,
            weight_decay=cfg.TRAIN.WEIGHT_DECAY,
            nesterov=cfg.TRAIN.NESTEROV
        )
    elif cfg.TRAIN.OPTIMIZER == 'adam':
        optimizer = torch.optim.Adam(
            model_parameters,
            lr=cfg.TRAIN.LR
        )

    return optimizer