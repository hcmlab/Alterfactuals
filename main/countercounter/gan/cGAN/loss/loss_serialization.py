from abc import ABCMeta, abstractmethod
from typing import List

import numpy as np

from main.countercounter.gan._execution.execution_utils.LossItems import DLossItems, GDCSLossItems


class LossPrinter(metaclass=ABCMeta):

    @abstractmethod
    def initialize(self, losses_discriminator, losses_generator) -> None:
        pass

    @abstractmethod
    def print(self):
        pass


class LossPrinterPPGAN(LossPrinter):

    def initialize(self, losses_discriminator: List[DLossItems], losses_generator: List[GDCSLossItems], losses_plausibility: List[DLossItems] = None):
        self._d_mean = np.mean(list(map(lambda loss: loss.overall, losses_discriminator)))
        self._g_overall_mean = np.mean(list(map(lambda loss: loss.overall, losses_generator)))
        self._g_discriminator_mean = np.mean(list(map(lambda loss: loss.discriminator, losses_generator)))
        self._g_classifier_mean = np.mean(list(map(lambda loss: loss.classifier, losses_generator)))
        self._g_ssim_mean = np.mean(list(map(lambda loss: loss.ssim, losses_generator)))
        self._g_autoencoder_mean = np.mean(list(map(lambda loss: loss.autoencoder, losses_generator)))
        self._g_kde_mean = np.mean(list(map(lambda loss: loss.kde, losses_generator)))

        self._csae_mean = np.mean(list(map(lambda loss: loss.csae, losses_generator)))
        self._svm_mean = np.mean(list(map(lambda loss: loss.svm, losses_generator)))

        if losses_plausibility is not None:
            self._d_plausibility_mean = np.mean(list(map(lambda loss: loss.overall, losses_plausibility)))
            self._g_plausibility_mean = np.mean(list(map(lambda loss: loss.plausibility, losses_generator)))
        else:
            self._d_plausibility_mean = 0
            self._g_plausibility_mean = 0

    def print(self):
        dual_print = f"Plaus Gen {self._g_plausibility_mean:.6f} - Plausibility loss D {self._d_plausibility_mean:.6f}"

        return f'G overall {self._g_overall_mean:.6f} disc {self._g_discriminator_mean:.6f} classif {self._g_classifier_mean} ssim {self._g_ssim_mean} svm {self._svm_mean} csae {self._csae_mean} {dual_print} kde {self._g_kde_mean} - Discriminator loss {self._d_mean:.6f}'


class LossPrinterPix2Pix(LossPrinterPPGAN):

    def __init__(self, use_classifier=True, dual=False):
        self._use_classifier = use_classifier
        self.dual = dual

    def initialize(self, losses_discriminator: List[DLossItems], losses_generator: List[GDCSLossItems], losses_plausibility: List[DLossItems]):
        if self._use_classifier:
            if self.dual:
                return super().initialize(losses_discriminator, losses_generator, losses_plausibility)
            else:
                return super().initialize(losses_discriminator, losses_generator, None)

        self._d_mean = np.mean(list(map(lambda loss: loss.overall, losses_discriminator)))
        self._g_overall_mean = np.mean(list(map(lambda loss: loss.overall, losses_generator)))
        self._g_discriminator_mean = np.mean(list(map(lambda loss: loss.discriminator, losses_generator)))
        self._g_ssim_mean = np.mean(list(map(lambda loss: loss.ssim, losses_generator)))

    def print(self):
        if self._use_classifier:
            return super().print()

        return f'G overall {self._g_overall_mean:.6f} disc {self._g_discriminator_mean:.6f} ssim {self._g_ssim_mean} - Discriminator loss {self._d_mean:.6f}'


class LossTensorboardLogger(metaclass=ABCMeta):

    @abstractmethod
    def initialize(self, losses_discriminator, losses_generator):
        pass

    @property
    @abstractmethod
    def loggable_properties(self):
        pass


class LossTensorboardLoggerPPGAN(LossTensorboardLogger):

    def initialize(self, losses_discriminator: List[DLossItems], losses_generator: List[GDCSLossItems]):
        self._d_mean = np.mean(list(map(lambda loss: loss.overall, losses_discriminator)))
        self._g_overall_mean = np.mean(list(map(lambda loss: loss.overall, losses_generator)))
        self._g_discriminator_mean = np.mean(list(map(lambda loss: loss.discriminator, losses_generator)))
        self._g_verificator_mean = np.mean(list(map(lambda loss: loss.classifier, losses_generator)))
        self._g_ssim_mean = np.mean(list(map(lambda loss: loss.ssim, losses_generator)))
        self._g_autoencoder_mean = np.mean(list(map(lambda loss: loss.autoencoder, losses_generator)))
        self._g_kde_mean = np.mean(list(map(lambda loss: loss.kde, losses_generator)))
        self._csae_mean = np.mean(list(map(lambda loss: loss.csae, losses_generator)))
        self._svm_mean = np.mean(list(map(lambda loss: loss.svm, losses_generator)))

    @property
    def loggable_properties(self):
        return [
            ('training generator loss overall', self._g_overall_mean),
            ('training generator loss discriminator', self._g_discriminator_mean),
            ('training generator loss classifier', self._g_verificator_mean),
            ('training generator loss ssim', self._g_ssim_mean),
            ('training generator loss ae', self._g_autoencoder_mean),
            ('training generator loss kde', self._g_kde_mean),
            ('training generator loss svm', self._svm_mean),
            ('training generator loss csae', self._csae_mean),
            ('training discriminator loss', self._d_mean),
        ]


class LossTensorboardLoggerPix2Pix(LossTensorboardLoggerPPGAN):

    def __init__(self, use_classifier=True, dual=False):
        self._use_classifier = use_classifier
        self.dual = dual

    def initialize(self, losses_discriminator: List[DLossItems], losses_generator: List[GDCSLossItems], losses_plausibility: List[DLossItems]):
        if self._use_classifier:
            super().initialize(losses_discriminator, losses_generator)
            return

        self._d_mean = np.mean(list(map(lambda loss: loss.overall, losses_discriminator)))
        self._g_overall_mean = np.mean(list(map(lambda loss: loss.overall, losses_generator)))
        self._g_discriminator_mean = np.mean(list(map(lambda loss: loss.discriminator, losses_generator)))
        self._g_ssim_mean = np.mean(list(map(lambda loss: loss.ssim, losses_generator)))

        if self.dual:
            self._d_plausibility_mean = np.mean(list(map(lambda loss: loss.overall, losses_plausibility)))
            self._g_plausibility_mean = np.mean(list(map(lambda loss: loss.plausibility, losses_generator)))

    @property
    def loggable_properties(self):
        if self._use_classifier:
            return super().loggable_properties
        result = [
            ('training generator loss overall', self._g_overall_mean),
            ('training generator loss discriminator', self._g_discriminator_mean),
            ('training generator loss ssim', self._g_ssim_mean),
            ('training discriminator loss', self._d_mean),
        ]

        if self.dual:
            result.append(
                ('training generator loss plausibility discriminator', self._g_plausibility_mean)
            )
            result.append(
                ('training plausibility discriminator loss', self._d_plausibility_mean)
            )