import time

import torch

from .utils.meters import AverageMeter
from .loss import MMCL
from .labelprediction import MPLP

class Trainer(object):
    def __init__(self, cfg, model, memory):
        super(Trainer, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model
        self.memory = memory
        self.labelpred = MPLP(cfg.MPLP.T)
        self.criterion = MMCL(cfg.MMCL.DELTA, cfg.MMCL.R).to(self.device)

    def train(self, epoch, data_loader, optimizer, print_freq=1):
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()

        end = time.time()
        for i, inputs in enumerate(data_loader):
            data_time.update(time.time() - end)

            inputs, pids = self._parse_data(inputs)
            outputs = self.model(inputs, 'l2feat')
            logits = self.memory(outputs, pids, epoch)

            if epoch > 5:
                multilabel = self.labelpred.predict(self.memory.mem.detach().clone(), pids.detach().clone())
                loss = self.criterion(logits, multilabel, True)
            else:
                loss = self.criterion(logits, pids)

            losses.update(loss.item(), outputs.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                log = "Epoch: [{}][{}/{}], Time {:.3f} ({:.3f}), Data {:.3f} ({:.3f}), Loss {:.3f} ({:.3f})" \
                    .format(epoch, i + 1, len(data_loader),
                            batch_time.val, batch_time.avg,
                            data_time.val, data_time.avg,
                            losses.val, losses.avg)
                print(log)

    def _parse_data(self, inputs):
        imgs, _, _, pids = inputs
        inputs = imgs.to(self.device)
        pids = pids.to(self.device)
        return inputs, pids