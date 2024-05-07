from abc import abstractmethod, ABCMeta

import torch

from main.countercounter.gan._execution.execution_utils.LossItems import DLossItems
from main.countercounter.gan.cGAN.loss.generator_loss_calculator import LossCalculator
from main.countercounter.gan.utils.label_maker import get_real_labels, get_fake_labels


class DLossCalculator(LossCalculator, metaclass=ABCMeta):

    def __init__(self, device):
        super().__init__(device)

    @abstractmethod
    def calculate(self, real_prediction, fake_prediction):
        pass


class DLossCalculatorPPGAN(DLossCalculator):

    def __init__(self, device, discriminator_loss, one_sided_label_smoothing):
        super().__init__(device)
        self.discriminator_loss = discriminator_loss
        self.one_sided_label_smoothing = one_sided_label_smoothing

    @abstractmethod
    def calculate(self, real_prediction, fake_prediction):
        real_labels = get_real_labels(real_prediction.size(), self.device, self.one_sided_label_smoothing)
        real_loss = self.discriminator_loss.calculate(real_prediction, real_labels)

        fake_labels = get_fake_labels(fake_prediction.size(), self.device)
        fake_loss = self.discriminator_loss.calculate(fake_prediction, fake_labels)

        self._loss = real_loss + fake_loss

        self._loss_item = self._loss.item()

    @property
    def loss_tensor(self):
        return self._loss

    @property
    def loss_stats(self):
        return DLossItems(self._loss_item)


class DLossCalculatorPatchGAN(DLossCalculator):

    def __init__(self, device, discriminator_loss, one_sided_label_smoothing, wgan=False):
        super().__init__(device)
        self.discriminator_loss = discriminator_loss
        self.one_sided_label_smoothing = one_sided_label_smoothing

        self.wgan = wgan

    def calculate(self, real_prediction, fake_prediction):
        self._loss = self._get_loss(fake_prediction, real_prediction, self.one_sided_label_smoothing, self.discriminator_loss)
        self._loss_item = self._loss.item()

    def _get_loss(self, fake_prediction, real_prediction, one_sided_label_smoothing, loss):
        real_labels = get_real_labels(real_prediction.size(), self.device, one_sided_label_smoothing)
        fake_labels = get_fake_labels(fake_prediction.size(), self.device)

        if not self.wgan:
            real_loss = loss.calculate(real_prediction, real_labels)

            fake_loss = loss.calculate(fake_prediction, fake_labels)
        else:
            real_loss = - torch.mean(real_prediction)
            fake_loss = torch.mean(fake_prediction)

        return 0.5 * (real_loss + fake_loss)

    @property
    def loss_tensor(self):
        return self._loss

    @property
    def loss_stats(self):
        return DLossItems(self._loss_item)


class DualDLossCalculatorPatchGAN(DLossCalculatorPatchGAN):

    def __init__(self, device, discriminator_loss, one_sided_label_smoothing, plausibility_discriminator_loss, one_sided_label_smoothing_plausibility, wgan=False):
        super().__init__(device, discriminator_loss, one_sided_label_smoothing, wgan)
        self.plausibility_discriminator_loss = plausibility_discriminator_loss
        self.one_sided_label_smoothing_plausibility = one_sided_label_smoothing_plausibility

    def calculate(self, real_prediction, fake_prediction, real_plausibility_prediction, fake_plausibility_prediction):
        super().calculate(real_prediction, fake_prediction)

        if real_plausibility_prediction is not None:
            self._plausibility_loss = self._get_loss(fake_plausibility_prediction, real_plausibility_prediction, self.one_sided_label_smoothing_plausibility, self.plausibility_discriminator_loss)
        else:
            self._plausibility_loss = torch.tensor(0)

        self._plausibility_loss_item = self._plausibility_loss.item()

    @property
    def loss_tensor(self):
        return self._loss

    @property
    def plausibility_loss_tensor(self):
        return self._plausibility_loss

    @property
    def loss_stats(self):
        return DLossItems(self._loss_item)

    @property
    def plausibility_loss_stats(self):
        return DLossItems(self._plausibility_loss_item)

    def reset_loss_tensor(self):
        super().reset_loss_tensor()
        self._plausibility_loss = None