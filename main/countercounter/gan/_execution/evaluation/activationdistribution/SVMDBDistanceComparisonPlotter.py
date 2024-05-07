from os.path import sep

import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")


class SVMDBDistanceComparisonPlotter:

    def __init__(self, distance_differences, output_folder):
        self.distance_differences = distance_differences

        self.output_folder = output_folder

    def plot(self):
        for name, distances in self.distance_differences:
            self._plot(name, distances)

    def _plot(self, name, distances):
        fig, ax = plt.subplots()

        mean = -1
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

            lines = [
                f'Mean in plot: {mean} (-1 is dummy value)\n',
                f'Std in plot: {std} (-1 is dummy value)\n'
            ]
            with open(f'{self.output_folder}{sep}mean_info_{name}.txt', 'w') as file:
                file.writelines(lines)

        except IndexError as e:
            pass

        plt.title(f'Differences between Original and Generated: Distances to Decision Boundary (normed) -- {name}')
        plt.xlabel('Distance')
        plt.savefig(f'{self.output_folder}/db_distances_normed_{name}.png', bbox_inches="tight")
        plt.close()

        sns.distplot(distances, kde=False)
        plt.title(f'Counts of Differences between Original and Generated: Distances to Decision Boundary (normed) -- {name}')
        plt.xlabel('Distance')
        plt.savefig(f'{self.output_folder}/db_distances_count_normed_{name}.png', bbox_inches="tight")
        plt.close()