import torch.nn
from piqa.ssim import SSIM
from torch import nn

from main.countercounter.classifier.dataset.DataLoader import denormalize
from main.countercounter.gan.ssim.SSIMLoss import SSIMLoss


class MSELossSimilarityEncouraged(nn.Module, SSIMLoss):

    def __init__(self, device):
        super().__init__()
        self._loss = torch.nn.MSELoss().to(device)

    def forward(self, x, x_hat):
        return self._loss(x, x_hat)
