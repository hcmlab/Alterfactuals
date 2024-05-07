import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")
import numpy as np


class SVMDBDistancePlotter:

    def __init__(self, name_dist_avg_std, output_folder):
        self.name_dist_avg_std = name_dist_avg_std

        self.output_folder = output_folder

    def plot(self):
        self._print()
        self._plot()

    def _print(self):
        lines = []

        for name, distances, avg, std in self.name_dist_avg_std:
            lines.append(
                f'Dataset: {name} -- Avg. distance to DB: {avg} -- Std. in DB distance: {std}\n'
            )

        with open(f'{self.output_folder}/svm_distance_results.txt', 'w') as file:
            file.writelines(lines)

    def _plot(self):
        for name, distances, avg, std in self.name_dist_avg_std:
            self._plot_distances(name, distances, avg, std)

    def _plot_distances(self, name, distances, mean, std):
        fig, ax = plt.subplots()

        sns.kdeplot(distances, shade=False, ax=ax)
        try:
            kdeline = ax.lines[0]
            xs = kdeline.get_xdata()
            ys = kdeline.get_ydata()

            left_std = mean - std
            right_std = mean + std

            ax.vlines(mean, 0, np.interp(mean, xs, ys), ls=':')
            ax.fill_between(xs, 0, ys, facecolor='blue', alpha=0.2)
            ax.fill_between(xs, 0, ys, where=(left_std <= xs) & (xs <= right_std), interpolate=True, facecolor='blue',
                            alpha=0.2)
        except IndexError as e:
            pass

        plt.title(f'Distances to Decision Boundary (normed) -- {name}')
        plt.xlabel('Distance')
        plt.savefig(f'{self.output_folder}/db_distances_{name}.png', bbox_inches="tight")
        plt.close()

        sns.distplot(distances, kde=False)
        plt.title(f'Counts of Distances to Decision Boundary (normed) -- {name}')
        plt.xlabel('Distance')
        plt.savefig(f'{self.output_folder}/db_distances_count_{name}.png', bbox_inches="tight")
        plt.close()