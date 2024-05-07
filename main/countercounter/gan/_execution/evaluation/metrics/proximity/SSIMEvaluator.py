import os
from os.path import sep

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")
import seaborn as sns


class SSIMEvaluator:

    def __init__(self, output_folder, ssim_csv, ssim_key=None):
        self.output_folder = output_folder
        self.ssim_csv = ssim_csv

        self.ssim_key = ssim_key if ssim_key is not None else 'SSIM'

    def evaluate(self):
        # columns = ['Filename1', 'Filename2', 'SSIM']
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        df = pd.read_csv(self.ssim_csv, sep=';')

        ssim_values = df[self.ssim_key].values.tolist()

        self._plot(ssim_values)
        self._print_stats(ssim_values)

    def _plot(self, ssim_values):
        bins = list(range(0, 101, 5))

        sns.histplot(ssim_values, bins=bins)
        plt.xlim([0, 1])

        plt.title(f'SSIM Values')
        plt.xlabel('SSIM')

        plt.savefig(f'{self.output_folder}{sep}ssim_values_count.png', bbox_inches="tight")
        plt.close()

        fig, ax = plt.subplots()
        sns.kdeplot(ssim_values, shade=False, ax=ax)

        try:
            kdeline = ax.lines[0]
            xs = kdeline.get_xdata()
            ys = kdeline.get_ydata()

            mean = np.mean(ssim_values)
            std = np.std(ssim_values)
            left_std = mean - std
            right_std = mean + std

            ax.vlines(mean, 0, np.interp(mean, xs, ys), ls=':')
            ax.fill_between(xs, 0, ys, facecolor='blue', alpha=0.2)
            ax.fill_between(xs, 0, ys, where=(left_std <= xs) & (xs <= right_std), interpolate=True, facecolor='blue',
                            alpha=0.2)
        except IndexError:
            pass

        plt.xlim([0, 1])

        plt.title(f'SSIM Values')
        plt.xlabel('SSIM')

        plt.savefig(f'{self.output_folder}{sep}ssim_values.png', bbox_inches="tight")
        plt.close()

    def _print_stats(self, ssim_values):
        avg = np.mean(ssim_values)
        std = np.std(ssim_values)
        lines = [
            'SSIM\n',
            f'Average: {avg}\n',
            f'Standard deviation: {std}\n'
        ]
        with open(f'{self.output_folder}{sep}ssim_stats.txt', 'w') as file:
            file.writelines(lines)