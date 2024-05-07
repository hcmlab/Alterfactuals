from os.path import sep

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")
from scipy import stats


class WeightActivationPlotter:

    def __init__(self, abs_weight_by_feature_by_class, abs_weight_by_feature, distances_by_feature_by_class, output_path, plot_zero_dist=False, weight_by_feature_by_class=None):
        self.abs_weight_by_feature_by_class = abs_weight_by_feature_by_class
        self.abs_weight_by_feature = abs_weight_by_feature
        self.weight_by_feature_by_class = weight_by_feature_by_class

        self.distances_by_feature_by_class = distances_by_feature_by_class

        self.output_path = output_path

        self.spearman_correlation = []  # (corr coefficient, p_value for 0-hypothesis "no correlation", descriptor of setting)

        self.plot_zero_dist = plot_zero_dist

    def plot(self):
        avg_diff_per_feature = self._get_average_dist_per_feature(self.distances_by_feature_by_class)

        # over both classes together
        self._plot('weight class 0 - all samples', self.combine(avg_diff_per_feature, self.abs_weight_by_feature_by_class, 0))
        self._plot('weight class 1 - all samples', self.combine(avg_diff_per_feature, self.abs_weight_by_feature_by_class, 1))
        self._plot('weight both classes - all samples', self.combine(avg_diff_per_feature, self.abs_weight_by_feature, None))

        avg_diff_per_feature_0 = self._get_average_dist_per_feature(self.distances_by_feature_by_class, 0)
        self._plot('weight class 0 - pairs of class 0', self.combine(avg_diff_per_feature_0, self.abs_weight_by_feature_by_class, 0))
        self._plot('weight class 1 - pairs of class 0', self.combine(avg_diff_per_feature_0, self.abs_weight_by_feature_by_class, 1))
        self._plot('weight both classes - pairs of class 0', self.combine(avg_diff_per_feature_0, self.abs_weight_by_feature, None))

        avg_diff_per_feature_1 = self._get_average_dist_per_feature(self.distances_by_feature_by_class, 1)
        self._plot('weight class 0 - pairs of class 1', self.combine(avg_diff_per_feature_1, self.abs_weight_by_feature_by_class, 0))
        self._plot('weight class 1 - pairs of class 1', self.combine(avg_diff_per_feature_1, self.abs_weight_by_feature_by_class, 1))
        self._plot('weight both classes - pairs of class 1', self.combine(avg_diff_per_feature_1, self.abs_weight_by_feature, None))


        # each neuron twice, once for each class + split diff
        self._plot('same class weights for each class', self.combine_per_class(avg_diff_per_feature_0, avg_diff_per_feature_1))
        self._plot('counter weight for each class', self.combine_per_class(avg_diff_per_feature_0, avg_diff_per_feature_1, counter=True))

        self._plot('avg diff - abs diff of weights', self.combine_with_actual_weights(avg_diff_per_feature, self.weight_by_feature_by_class))

        self._print_spearman()

    def _get_average_dist_per_feature(self, dict_to_average, label = None):
        avg_dist_per_feature = {}

        for feature in dict_to_average:
            items_to_average = dict_to_average[feature]

            if label is not None:
                try:
                    items_to_average = items_to_average[label]
                except KeyError:
                    items_to_average = []  # results in nan
            else:
                items = []
                for label in items_to_average.keys():
                    items.extend(items_to_average[label])
                items_to_average = items

            items_to_average = list(map(lambda i: i[0], items_to_average))
            avg_dist_per_feature[feature] = np.mean(items_to_average)

        return avg_dist_per_feature

    def combine(self, dict1_by_feature, dict2_by_feature_and_maybe_class, label):
        res1 = []
        res2 = []

        for feature in dict1_by_feature.keys():
            v1 = dict1_by_feature[feature]

            v2 = dict2_by_feature_and_maybe_class[feature]
            if label is not None:
                v2 = v2[label]

            if np.isnan(v1) or np.isnan(v2):
                continue

            if v1 == 0 and not self.plot_zero_dist: # weight is 0, and should not be plotted to make spotting of correlations easier
                continue

            res1.append(v1)
            res2.append(v2)

        return res1, res2

    def combine_per_class(self, avg_diff_per_feature_0, avg_diff_per_feature_1, counter=False):
        res1 = []
        res2 = []

        for feature in avg_diff_per_feature_0.keys():
            v1 = avg_diff_per_feature_0[feature]

            clazz = 0 if not counter else 1
            v2 = self.abs_weight_by_feature_by_class[feature][clazz]

            if np.isnan(v1) or np.isnan(v2):
                continue

            if v1 == 0 and not self.plot_zero_dist: # weight is 0, and should not be plotted to make spotting of correlations easier
                continue

            res1.append(v1)
            res2.append(v2)

        for feature in avg_diff_per_feature_1.keys():
            v1 = avg_diff_per_feature_1[feature]

            clazz = 1 if not counter else 0
            v2 = self.abs_weight_by_feature_by_class[feature][clazz]

            if np.isnan(v1) or np.isnan(v2):
                continue

            if v1 == 0 and not self.plot_zero_dist: # weight is 0, and should not be plotted to make spotting of correlations easier
                continue

            res1.append(v1)
            res2.append(v2)

        return res1, res2

    def _plot(self, descriptor, values):
        distances, weights = values
        if not distances or not weights:return
        weights, distances = zip(*sorted(zip(weights, distances)))

        plt.scatter(weights, distances)
        plt.title(f'Abs Weight vs. Abs Activation Distance - {descriptor}')
        plt.xlabel('Abs Weight')
        plt.ylabel('Abs Activation Distance')

        plt.savefig(f'{self.output_path}{sep}weight_activation_{descriptor}_scatter.png', bbox_inches="tight")
        plt.close()

        coeff, p_value = stats.spearmanr(weights, distances)

        self.spearman_correlation.append((coeff, p_value, descriptor))

    def _print_spearman(self):
        lines = [
            'Weights vs. activations analysed with the Spearman correlation\n',
            '\n',
            'Coefficient 0: no correlation, -1, +1 exact monotonic relationship\n',
            'A p-Value <= 0.05 means that zero hypothesis (no correlation) is false.\n',
            '(Technically, switching the hypotheses would be better ...)\n',
            'ATT: p-Value does not reveal how strong the relation is (--> coefficient).\n'
            '\n',
        ] + [
            f'{data[2]}: Coefficient: {data[0]} -- p-Value: {data[1]}\n' for data in self.spearman_correlation
        ]

        with open(f'{self.output_path}{sep}spearman_correlation.txt', 'w') as file:
            file.writelines(lines)

    def combine_with_actual_weights(self, avg_diff_per_feature, weight_by_feature_by_class):
        res1 = []
        res2 = []

        for feature in avg_diff_per_feature.keys():
            v1 = avg_diff_per_feature[feature]

            weight_by_class = weight_by_feature_by_class[feature]
            abs_diff_weights = np.abs(weight_by_class[0] - weight_by_class[1])

            if np.isnan(v1) or np.isnan(abs_diff_weights):
                continue

            if v1 == 0 and weight_by_class[0] == 0 and weight_by_class[1] == 0 and not self.plot_zero_dist: # weight is 0, and should not be plotted to make spotting of correlations easier
                continue

            res1.append(v1)
            res2.append(abs_diff_weights)

        return res1, res2