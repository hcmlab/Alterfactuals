from abc import abstractmethod, ABCMeta

import torch

from main.countercounter.gan._execution.execution_utils.LossItems import GDCSLossItems, DualGDCSLossItems
from main.countercounter.gan.autoencoder.Autoencoder import Autoencoder
from main.countercounter.gan.classifier.Classifier import Classifier
from main.countercounter.gan.svm.SVM import SVM
from main.countercounter.gan.utils.label_maker import get_real_labels


class LossCalculator(metaclass=ABCMeta):

    def __init__(self, device):
        self.device = device

    @property
    @abstractmethod
    def loss_tensor(self):
        pass

    @property
    @abstractmethod
    def loss_stats(self):
        pass

    def reset_loss_tensor(self):
        self._discriminator_loss = None


class GLossCalculator(LossCalculator, metaclass=ABCMeta):

    def __init__(self, device):
        super().__init__(device)

    @abstractmethod
    def calculate(self, real_data, generated_data, discriminator_judgement, class_label):
        pass

    def _get_labmda_ssim_by_epoch(self, lambda_ssim, ssim_epoch_escalation, escalation_by_epoch, epoch):
        if not ssim_epoch_escalation:
            return lambda_ssim

        return lambda_ssim + escalation_by_epoch * epoch


class GLossCalculatorCounter(GLossCalculator):

    def __init__(self, device, generator_loss, classifier: Classifier, ssim, autoencoder: Autoencoder, kde, lambda_classifier, lambda_ssim, lambda_autoencoder, lambda_kde, one_sided_label_smoothing, wgan=False, ssim_epoch_escalation=False, escalation_by_epoch=0.01, lambda_csae=0, csae=None, svm=None, lambda_svm=0):
        super().__init__(device)
        self.generator_loss = generator_loss
        self.classifier = classifier
        self.ssim = ssim
        self.autoencoder = autoencoder
        self.kde = kde
        self.csae = csae
        self.lambda_classifier = lambda_classifier
        self.lambda_ssim = lambda_ssim
        self.lambda_autoencoder = lambda_autoencoder
        self.lambda_kde = lambda_kde
        self.lambda_csae = lambda_csae
        self.lambda_svm = lambda_svm
        self.one_sided_label_smoothing = one_sided_label_smoothing

        self.ssim_epoch_escalation = ssim_epoch_escalation
        self.escalation_by_epoch = escalation_by_epoch
        self.wgan = wgan

        self.svm: SVM = svm

    def calculate(self, real_data, generated_data, discriminator_judgement, class_label, epoch, activations_real, activations_generated):
        labels = get_real_labels(discriminator_judgement.size(), self.device, self.one_sided_label_smoothing)

        if not self.wgan:
            self._discriminator_loss = self.generator_loss.calculate(discriminator_judgement, labels)
        else:
            self._discriminator_loss = - torch.mean(discriminator_judgement)

        self._classifier_loss = self.classifier(real_data, generated_data)
        self._classifier_loss.requires_grad = True
        self._ssim_loss = self.ssim(real_data, generated_data)

        self._csae_loss = self.csae(real_data, class_label, generated_data, class_label)
        self._csae_loss.requires_grad = True

        lambda_ssim = self._get_labmda_ssim_by_epoch(self.lambda_ssim, self.ssim_epoch_escalation, self.escalation_by_epoch, epoch)

        if self.autoencoder is not None:
            self._autoencoder_loss = self.autoencoder(generated_data)
            self._autoencoder_loss.requires_grad = True
        else:
            self._autoencoder_loss = torch.tensor(0.)

        self._kde_loss = self.kde(generated_data, class_label).float()
        self._kde_loss.requires_grad = True

        if self.svm is not None:
            self._svm_loss = self.svm.get_loss(activations_real, activations_generated)
            self._svm_loss.requires_grad = True
        else:
            self._svm_loss = torch.tensor(0.)

        self._loss = self._discriminator_loss + self.lambda_classifier * self._classifier_loss + lambda_ssim * self._ssim_loss + self.lambda_csae * self._csae_loss + self.lambda_svm * self._svm_loss

        self._overall_loss_item = self._loss.item()
        self._discriminator_loss_item = self._discriminator_loss.item()
        self._classifier_loss_item = self._classifier_loss.item()
        self._ssim_loss_item = self._ssim_loss.item()
        self._autoencoder_loss_item = self._autoencoder_loss.item()
        self._kde_loss_item = self._kde_loss.item()
        self._csae_loss_item = self._csae_loss.item()
        self._svm_loss_item = self._svm_loss.item()

    def reset_loss_tensor(self):
        super().reset_loss_tensor()

        self._discriminator_loss = None
        self._classifier_loss = None
        self._ssim_loss = None
        self._autoencoder_loss = None
        self._kde_loss = None
        self._csae_loss = None
        self._svm_loss = None

    @property
    def loss_tensor(self):
        return self._loss

    @property
    def loss_stats(self):
        return GDCSLossItems(
            overall=self._overall_loss_item,
            discriminator=self._discriminator_loss_item,
            classifier=self._classifier_loss_item,
            ssim=self._ssim_loss_item,
            autoencoder=self._autoencoder_loss_item,
            kde=self._kde_loss_item,
            csae=self._csae_loss_item,
            svm=self._svm_loss_item,
        )


class DualGLossCalculatorCounter(GLossCalculator):

    def __init__(self, device, generator_loss, plausibility_generator_loss, classifier: Classifier, ssim, autoencoder: Autoencoder, kde, lambda_classifier, lambda_ssim, lambda_autoencoder, lambda_kde, one_sided_label_smoothing, lambda_plausibility, one_sided_label_smoothing_plausibility, wgan=False, ssim_epoch_escalation=False, escalation_by_epoch=0.01, lambda_csae=0, csae=None, svm=None, lambda_svm=0):
        super().__init__(device)
        self.generator_loss = generator_loss
        self.plausibility_generator_loss = plausibility_generator_loss
        self.classifier = classifier
        self.ssim = ssim
        self.autoencoder = autoencoder
        self.kde = kde
        self.csae = csae
        self.lambda_classifier = lambda_classifier
        self.lambda_ssim = lambda_ssim
        self.lambda_autoencoder = lambda_autoencoder
        self.lambda_kde = lambda_kde
        self.lambda_csae = lambda_csae
        self.lambda_svm = lambda_svm
        self.one_sided_label_smoothing = one_sided_label_smoothing

        self.lambda_plausibility = lambda_plausibility
        self.one_sided_label_smoothing_plausibility = one_sided_label_smoothing_plausibility

        self.ssim_epoch_escalation = ssim_epoch_escalation
        self.escalation_by_epoch = escalation_by_epoch
        self.wgan = wgan

        self.svm = svm

    def calculate(self, real_data, generated_data, discriminator_judgement, class_label, plausibility_judgement, epoch, activations_real, activations_generated):
        labels = get_real_labels(discriminator_judgement.size(), self.device, self.one_sided_label_smoothing)

        if not self.wgan:
            self._discriminator_loss = self.generator_loss.calculate(discriminator_judgement, labels)
        else:
            self._discriminator_loss = - torch.mean(discriminator_judgement)

        if plausibility_judgement is not None:
            plausibility_labels = get_real_labels(plausibility_judgement.size(), self.device, self.one_sided_label_smoothing_plausibility)
            self._plausibility_loss = self.plausibility_generator_loss.calculate(plausibility_judgement, plausibility_labels)
        else:
            self._plausibility_loss = torch.tensor(0)

        self._classifier_loss = self.classifier(real_data, generated_data)
        self._classifier_loss.requires_grad = True
        self._ssim_loss = self.ssim(real_data, generated_data)

        self._csae_loss = self.csae(real_data, class_label, generated_data, class_label)
        self._csae_loss.requires_grad = True

        lambda_ssim = self._get_labmda_ssim_by_epoch(self.lambda_ssim, self.ssim_epoch_escalation, self.escalation_by_epoch, epoch)

        if self.autoencoder is not None:
            self._autoencoder_loss = self.autoencoder(generated_data)
            self._autoencoder_loss.requires_grad = True
        else:
            self._autoencoder_loss = torch.tensor(0)

        self._kde_loss = self.kde(generated_data, class_label).float()
        self._kde_loss.requires_grad = True

        if self.svm is not None:
            self._svm_loss = self.svm.get_loss(activations_real, activations_generated)
        else:
            self._svm_loss = torch.tensor(0)

        self._loss = self._discriminator_loss + self.lambda_classifier * self._classifier_loss + lambda_ssim * self._ssim_loss + self.lambda_plausibility * self._plausibility_loss + self.lambda_csae * self._csae_loss + self.lambda_svm * self._svm_loss

        self._overall_loss_item = self._loss.item()
        self._discriminator_loss_item = self._discriminator_loss.item()
        self._plausibility_loss_item = self._plausibility_loss.item()
        self._classifier_loss_item = self._classifier_loss.item()
        self._ssim_loss_item = self._ssim_loss.item()
        self._autoencoder_loss_item = self._autoencoder_loss.item()
        self._kde_loss_item = self._kde_loss.item()
        self._csae_loss_item = self._csae_loss.item()
        self._svm_loss_item = self._svm_loss.item()

    def reset_loss_tensor(self):
        super().reset_loss_tensor()

        self._discriminator_loss = None
        self._plausibility_loss = None
        self._classifier_loss = None
        self._ssim_loss = None
        self._autoencoder_loss = None
        self._kde_loss = None
        self._csae_loss = None
        self._svm_loss = None

    @property
    def loss_tensor(self):
        return self._loss

    @property
    def loss_stats(self):
        return DualGDCSLossItems(
            overall=self._overall_loss_item,
            discriminator=self._discriminator_loss_item,
            plausibility=self._plausibility_loss_item,
            classifier=self._classifier_loss_item,
            ssim=self._ssim_loss_item,
            autoencoder=self._autoencoder_loss_item,
            kde=self._kde_loss_item,
            csae=self._csae_loss_item,
            svm=self._svm_loss_item,
        )