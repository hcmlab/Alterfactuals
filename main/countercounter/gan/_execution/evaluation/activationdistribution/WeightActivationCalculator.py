from collections import defaultdict

import torch
import numpy as np


class WeightActivationCalculator:

    def __init__(self, feature_count, cnn):
        self.feature_count = feature_count

        self.cnn = cnn

    def calculate(self):
        classifier_weights = self.cnn.classifier.weight.cpu().detach()
        abs_weights = torch.abs(classifier_weights)

        abs_weight_by_feature_by_class = defaultdict(dict)
        abs_weight_by_feature = {}
        weight_by_feature_by_class = defaultdict(dict)

        for feature in range(self.feature_count):
            weight_class_0 = abs_weights[0][feature].item()
            weight_class_1 = abs_weights[1][feature].item()

            abs_weight_by_feature_by_class[feature][0] = weight_class_0
            abs_weight_by_feature_by_class[feature][1] = weight_class_1

            abs_weight_by_feature[feature] = np.mean([weight_class_0, weight_class_1])

            weight_by_feature_by_class[feature][0] = classifier_weights[0][feature].item()
            weight_by_feature_by_class[feature][1] = classifier_weights[1][feature].item()

        return abs_weight_by_feature_by_class, abs_weight_by_feature, weight_by_feature_by_class
