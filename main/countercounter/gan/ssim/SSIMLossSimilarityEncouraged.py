from piqa.ssim import SSIM
from torch import nn

from main.countercounter.classifier.dataset.DataLoader import denormalize
from main.countercounter.gan.ssim.SSIMLoss import SSIMLoss


class SSIMLossSimilarityEncouraged(nn.Module, SSIMLoss):

    def __init__(self, data_range, device, n_channels):
        super().__init__()
        self._ssim = SSIM(value_range=data_range, n_channels=n_channels, non_negative=True).to(device)

    def forward(self, x, x_hat):
        x = denormalize(x)
        x_hat = denormalize(x_hat)

        # self.assert_gte_0(x)
        # self.assert_le_1(x)
        #
        # self.assert_gte_0(x_hat)
        # self.assert_le_1(x_hat)

        return 1 - self.ssim(x, x_hat)

    def ssim(self, x, x_hat):
        return self._ssim(x, x_hat)