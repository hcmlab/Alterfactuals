import torch.nn as nn


class DiscriminatorLoss:

    def __init__(self, device):
        self.loss = nn.BCELoss().to(device)

    def calculate(self, prediction, labels):
        return self.loss(prediction, labels)


class GeneratorLoss:

    def __init__(self, device):
        self.loss = nn.BCELoss().to(device)

    def calculate(self, prediction, labels):
        return self.loss(prediction, labels)