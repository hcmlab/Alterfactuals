import torch
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")

import numpy as np
import pandas as pd

from main.countercounter.classifier.emotion_classifier.CustomNets import SoftmaxLogitWrapper
from main.countercounter.gan.utils.AbstractTraining import DEVICE


class Evaluator:

    def __init__(self, setup, config_nr, epoch):
        self.val_loader = setup.val_loader
        self.test_loader = setup.test_loader

        self.model = setup.model
        self.model.eval()

        self.scaled_model = setup.scaled_model
        if self.scaled_model is not None:
            self.scaled_model.eval()

        self.config_nr = config_nr
        self.epoch = epoch

    def evaluate(self):
        self._eval(self.test_loader, self.model, 'test', 'original')

    def _eval(self, data_loader, model, name, model_type):
        y_true = []
        y_pred = []

        total_preds = []
        db_distances = []
        # filenames = []

        logits = []

        inner_model = None
        if isinstance(self.model, SoftmaxLogitWrapper):
            inner_model = self.model.model
        for data, labels in data_loader:
            data = data.to(DEVICE)
            labels = labels.to(DEVICE)
            if len(labels.size()) == 1:
                labels_reshaped = labels.reshape(labels.size(0), 1)
            else:
                labels_reshaped = labels

            preds = model(data).cpu().detach()

            if inner_model is not None:
                logit, _ = inner_model(data)
                logits.append(logit.cpu().detach())

            if preds.size(1) > 1:
                _, classes = torch.max(preds, 1)
                _, labels = torch.max(labels_reshaped, 1)
            else:
                classes = (preds >= 0.5).float()
                classes = classes.reshape(classes.size(0))

            total_preds.append(preds.cpu().detach())
            db_distances.append(abs(0.5 - preds).cpu().detach())

            label_array = labels.cpu().detach().numpy()
            classes_array = classes.cpu().detach().numpy()

            y_true.extend(label_array)
            y_pred.extend(classes_array)

        print(f'{name} acc: {accuracy_score(y_true, y_pred):.6f}')
        print(f'{name} prec: {precision_score(y_true, y_pred):.6f}')
        print(f'{name} rec: {recall_score(y_true, y_pred):.6f}')
        print(f'{name} f1: {f1_score(y_true, y_pred):.6f}')
