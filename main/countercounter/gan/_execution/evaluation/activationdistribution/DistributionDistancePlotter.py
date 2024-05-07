import os
from os.path import sep
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")
import seaborn as sns
import numpy as np


class DistributionDistancePlotter:

    def __init__(self, distances_by_feature_by_class, total_distances, path_to_output_folder, additional_total_distances=None):
        self.distances_by_feature_by_class = distances_by_feature_by_class
        self.total_distances = total_distances
        self.path = path_to_output_folder

        self.additional_total_distances = additional_total_distances

        self.name = 'feature'

    def plot(self):
        for feature in self.distances_by_feature_by_class.keys():
            all_distances_per_feature = []

            for label in self.distances_by_feature_by_class[feature].keys():
                distances = list(map(lambda d: d[0], self.distances_by_feature_by_class[feature][label])) # list of (dist, cdf_test, cdf_gen)

                self._plot_dist(
                    f'of distances between test and generated image based on CDF - {self.name} {feature} class {label}',
                    'Distance',
                    f'cdf_distance_{self.name}_{feature}_class_{label}',
                    distances,
                )

                all_distances_per_feature.extend(distances)

            self._plot_dist(
                f'of distances between test and generated image based on CDF - {self.name} {feature} all classes',
                'Distance',
                f'cdf_distance_{self.name}_{feature}_all_classes',
                all_distances_per_feature,
            )

        self._plot_dist(
            'of total distances between test and generated image based on CDF',
            'Total distance',
            f'cdf_distance_{self.name}_total',
            self.total_distances,
            print_mean=True,
        )

        if self.additional_total_distances is not None:
            self._plot_dist(
                'of total distances between test and generated image based on CDF - same calc as weighted',
                'Total distance',
                'cdf_distance_total_same_calc_as_weighted',
                self.additional_total_distances,
                print_mean=True
            )

    def _plot_dist(self, title, x_label, filename, distances, print_mean=False):
        try:
            plt.close()
            fig, ax = plt.subplots()
            sns.kdeplot(distances, shade=False, ax=ax)

            try:
                kdeline = ax.lines[0]
                xs = kdeline.get_xdata()
                ys = kdeline.get_ydata()

                mean = np.mean(distances)
                std = np.std(distances)
                left_std = mean - std
                right_std = mean + std

                ax.vlines(mean, 0, np.interp(mean, xs, ys), ls=':')
                ax.fill_between(xs, 0, ys, facecolor='blue', alpha=0.2)
                ax.fill_between(xs, 0, ys, where=(left_std <= xs) & (xs <= right_std), interpolate=True, facecolor='blue',
                                alpha=0.2)

                if print_mean:
                    lines = [
                        f'Mean in plot: {mean:.6f}\n',
                        f'Std in plot: {std:.6f}\n',
                    ]
                    with open(f'{self.path}{sep}{filename}_mean_info.txt', 'w') as file:
                        file.writelines(lines)

            except IndexError as e:
                pass

            plt.title(f'Distribution {title}')
            plt.xlabel(x_label)

            plt.savefig(f'{self.path}{sep}{filename}_distribution.png', bbox_inches="tight")
            plt.close()

            sns.distplot(distances, kde=False)

            plt.title(f'Counts {title}')
            plt.xlabel(x_label)

            plt.savefig(f'{self.path}{sep}{filename}_count.png', bbox_inches="tight")
            plt.close()
        except:
            print('-------------------------------------------------------------')
            print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
            print('Skipping plot')
            print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
            print('-------------------------------------------------------------')
