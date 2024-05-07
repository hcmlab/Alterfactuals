import numpy as np
import torch
from joblib import load


class SVM:

    def __init__(self, path_to_svm):
        self._svm = self._load(path_to_svm)
        self.relevant_columns = None

    def get_loss(self, activations_real, activations_generated):
        if self.relevant_columns is None:
            self.relevant_columns = sorted([col for col in activations_real if col.startswith('feature')])

        real_acts = activations_real[self.relevant_columns]
        generated_acts = activations_generated[self.relevant_columns]

        real_proportional_distance = self._svm.decision_function(real_acts)
        generated_proportional_distance = self._svm.decision_function(generated_acts)

        sign_real = np.sign(real_proportional_distance)
        sign_gen = np.sign(generated_proportional_distance)

        if sign_real == sign_gen:
            raw_distances = abs(abs(np.array(real_proportional_distance)) - abs(np.array(generated_proportional_distance)))
        else:
            raw_distances = abs(np.array(real_proportional_distance) - np.array(generated_proportional_distance))

        return torch.tensor(raw_distances[0])

    def _load(self, path_to_svm):
        svm = load(path_to_svm)
        return svm