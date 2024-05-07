from abc import ABCMeta


class Loss(metaclass=ABCMeta):
    pass


class GDCSLossItems(Loss):

    def __init__(self, overall, discriminator, classifier, ssim, autoencoder, kde, csae, svm):
        self.overall = overall
        self.discriminator = discriminator
        self.classifier = classifier
        self.ssim = ssim
        self.autoencoder = autoencoder
        self.kde = kde
        self.csae = csae
        self.svm = svm


class DualGDCSLossItems(Loss):

    def __init__(self, overall, discriminator, plausibility, classifier, ssim, autoencoder, kde, csae, svm):
        self.overall = overall
        self.discriminator = discriminator
        self.plausibility = plausibility
        self.classifier = classifier
        self.ssim = ssim
        self.autoencoder = autoencoder
        self.kde = kde
        self.csae = csae
        self.svm = svm


class GDLossItems(Loss):

    def __init__(self, discriminator):
        self.discriminator = discriminator


class GDSLossItems(Loss):

    def __init__(self, overall, discriminator, ssim):
        self.overall = overall
        self.discriminator = discriminator
        self.ssim = ssim


class DLossItems(Loss):

    def __init__(self, overall):
        self.overall = overall