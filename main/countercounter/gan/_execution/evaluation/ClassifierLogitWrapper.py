import torch
from torch import nn


# model outputs one scalar, but I need probabilities for each class
class ClassifierLogitWrapper(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def _predict(self, x):
        return self.model.pred(x)

    def forward(self, x):
        y = self._predict(x)

        # e.g. output 0.6
        # meaning: 0.4% class 0, 0.6% class 1

        rest = 1 - y
        a = torch.cat([rest, y], dim=1)
        return a


class ClassifierLogitWithGradWrapper(ClassifierLogitWrapper):

    def _predict(self, x):
        return self.model.pred_with_grad(x)