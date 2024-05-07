from os.path import sep

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")

import numpy as np


class DistributionPlotter:

    def __init__(self, path_to_output_folder):
        self.path_to_output_folder = path_to_output_folder
        self.name = 'Feature'
        self.type = 'Activation'

    def _print_distributions(self, class_0_feature_values, class_1_feature_values, feature_idx):
        self._print_single_distribution(class_0_feature_values, '0', feature_idx)
        self._print_single_distribution(class_1_feature_values, '1', feature_idx)

        self._print_joint_distribution(
            [class_0_feature_values, class_1_feature_values],
            ['Class 0', 'Class 1'],
            f'{self.name} {feature_idx} {self.type} Comparison'
        )

    def _print_single_distribution(self, feature_values_of_class, class_name, feature_idx):
        plt.close()
        fig, ax = plt.subplots()

        sns.kdeplot(feature_values_of_class, shade=False, ax=ax)
        try:
            kdeline = ax.lines[0]
            xs = kdeline.get_xdata()
            ys = kdeline.get_ydata()

            mean = np.mean(feature_values_of_class)
            std = np.std(feature_values_of_class)
            left_std = mean - std
            right_std = mean + std

            ax.vlines(mean, 0, np.interp(mean, xs, ys), ls=':')
            ax.fill_between(xs, 0, ys, facecolor='blue', alpha=0.2)
            ax.fill_between(xs, 0, ys, where=(left_std <= xs) & (xs <= right_std), interpolate=True, facecolor='blue', alpha=0.2)
        except IndexError as e:
            pass

        plt.title(f'Distribution of {self.name} {feature_idx} for Class {class_name}')
        plt.xlabel(f'{self.name} {self.type} value')
        plt.savefig(f'{self.path_to_output_folder}{sep}{self.name}_{feature_idx}_class_{class_name}_distribution.png', bbox_inches="tight")
        plt.close()

        sns.distplot(feature_values_of_class, kde=False)
        plt.title(f'Counts of {self.name} {feature_idx} for Class {class_name}')
        plt.xlabel(f'{self.name} {self.type} value')
        plt.savefig(f'{self.path_to_output_folder}{sep}{self.name}_{feature_idx}_class_{class_name}_count.png', bbox_inches="tight")
        plt.close()

    def _print_joint_distribution(self, data, labels, name):
        colors = [
            'blue',
            'orange',
            'green',
            'red',
            'purple',
            'brown',
            'pink',
            'gray'
        ]

        fig, ax = plt.subplots()

        for idx, d in enumerate(data):
            sns.kdeplot(data=d, label=labels[idx], color=colors[idx], ax=ax)

            try:
                kdeline = ax.lines[idx]
                xs = kdeline.get_xdata()
                ys = kdeline.get_ydata()

                mean = np.mean(d)
                std = np.std(d)
                left_std = mean - std
                right_std = mean + std

                ax.vlines(mean, 0, np.interp(mean, xs, ys), ls=':')
                ax.fill_between(xs, 0, ys, facecolor=colors[idx], alpha=0.2)
                ax.fill_between(xs, 0, ys, where=(left_std <= xs) & (xs <= right_std), interpolate=True, facecolor=colors[idx],
                                alpha=0.2)
            except IndexError as e:
                pass

        plt.legend(loc='best')

        plt.title(f'{name}')
        plt.xlabel(f'{self.name} {self.type} value')
        plt.savefig(f'{self.path_to_output_folder}{sep}{name}.png', bbox_inches="tight")
        plt.close()

    def plot(self, class_0_feature_values, class_1_feature_values, feature):
        self._print_distributions(class_0_feature_values, class_1_feature_values, feature)