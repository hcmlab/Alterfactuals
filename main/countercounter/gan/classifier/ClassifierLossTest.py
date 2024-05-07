import torch
import numpy as np

from main.countercounter.gan.classifier.ClassifierLoss import InvertedArgmaxLoss, ArgmaxLoss

original_multi = torch.tensor(np.array([
    [0.13, 0.87],
    [0.51, 0.49],
]))

modified_multi = torch.tensor(np.array([
    [0.3, 0.7],
    [0.3, 0.7],
]))

original = torch.tensor([[0.2], [0.9]])
modified = torch.tensor([[0.6], [0.4]])


def test_classifier_loss_same() -> None:
    loss = InvertedArgmaxLoss()._distance(original, modified)
    loss = loss.reshape(loss.size(0))

    assert abs(loss[0].item() - 0.4) < 0.001
    assert abs(loss[1].item() - 0.4) < 0.001


def test_classifier_loss_not_same() -> None:
    loss = ArgmaxLoss(None)._distance(original, modified)
    loss = loss.reshape(loss.size(0))

    assert abs(loss[0].item() - 0.6) < 0.001
    assert abs(loss[1].item() - 0.6) < 0.001