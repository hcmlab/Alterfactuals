import numpy as np
import torch
from torchvision.utils import save_image

from main.countercounter.classifier.dataset.DataLoaderMNIST import denormalize
from main.countercounter.csvae.networks.CSVAE import CSVAE
from main.countercounter.gan.classifier.Classifier import Classifier


class Evaluator:

    def __init__(self, setup, config_nr, epoch):
        self.val_loader = setup.val_loader
        self.test_loader = setup.test_loader

        self.csvae: CSVAE = setup.csvae
        self.csvae.eval()
        self.classifier: Classifier = setup.classifier

        self.config_nr = config_nr
        self.epoch = epoch

    def evaluate(self):
        eval_accs = []

        for data, _ in self.test_loader:
            x = data
            y, y_scores = self.classifier.get_class_and_logits(x)

            x_pred, y_pred = self.csvae.get_x_pred_y_pred(x, y, y_scores)
            _, classes = torch.max(y_pred, 1)
            equals = (classes == y).cpu().type(torch.FloatTensor)
            eval_accs.append(torch.mean(equals).item())

        print(f'Accuracy on test data: {np.mean(eval_accs)}')

