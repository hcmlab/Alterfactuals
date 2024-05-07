from abc import ABCMeta
from datetime import datetime

import torch
from torch.utils.tensorboard import SummaryWriter

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class AbstractTraining(metaclass=ABCMeta):

    def __init__(self, tensorboard_dir, model_dir, checkpoints_dir, minimal_logging=False):
        self.device = DEVICE

        self.start_time = datetime.now()

        if not minimal_logging:
            self.writer = SummaryWriter(f'{tensorboard_dir}/run_{self.start_time:%Y_%m_%d_%H_%M}')
        self.model_dir = model_dir

        self.checkpoints_dir = checkpoints_dir