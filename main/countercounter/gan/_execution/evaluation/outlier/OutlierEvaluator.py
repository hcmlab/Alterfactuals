import itertools
from os.path import sep

from sklearn.neighbors import LocalOutlierFactor
import pandas as pd
from main.countercounter.classifier.dataset.DataLoaderMNIST import denormalize
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")
import seaborn as sns
import numpy as np
from scipy import stats


class OutlierEvaluator:

    def __init__(
            self,
            setup,
            setup_for_generated_images,
            path_to_output_folder='',
            path_to_train_ssim_distances=None,
    ):
        self.train_loader = setup.train_loader
        self.original_test_loader = setup.test_loader
        self.generated_test_loader = setup_for_generated_images.test_loader

        self.path_to_output_folder = path_to_output_folder
        self.raw_ssim = setup.ssim_function

        self.path_to_train_ssim_distances = path_to_train_ssim_distances

        self.train_images_by_filename = {}
        for train_image, _, filename in self.train_loader:
            self.train_images_by_filename[filename[0]] = train_image.cpu().detach()

    def evaluate(self):
        print(f'Evaluating:')
        print('---------------------------------------------------------')

        train_ssim_distances = self._load_train_ssim_distances()
        clf = self._fit_clustering(train_ssim_distances)
        print('Clustering on training set calculated')

        test_images, generated_images, filenames = self._get_images()
        test_train_ssim_distances, generated_train_ssim_distances = self._get_distances(test_images, generated_images)

        test_outlier_factors = self._get_outlier_factors(test_train_ssim_distances, clf)
        generated_outlier_factors = self._get_outlier_factors(generated_train_ssim_distances, clf)

        self._plot_distribution(test_outlier_factors, 'Test')
        self._plot_distribution(generated_outlier_factors, 'Generated')
        p_value = self._statistical_sameness_test(test_outlier_factors, generated_outlier_factors)
        wasserstein_dist = self._wasserstein_distance(test_outlier_factors, generated_outlier_factors)

        factor_diff = self._get_diff(test_outlier_factors, generated_outlier_factors)
        self._plot_factor_diff_distribution(factor_diff)
        mean, std = self._mean_std(factor_diff)
        test_mean, test_std = self._mean_std(test_outlier_factors)
        gen_mean, gen_std = self._mean_std(generated_outlier_factors)
        self._store(test_outlier_factors, generated_outlier_factors, factor_diff, filenames)
        coeff, p_value_spearman = self._LOF_correlation(test_outlier_factors, factor_diff)

        test_count, gen_count, both_count = self._get_outlier_count(test_train_ssim_distances, generated_train_ssim_distances, clf)

        self._store_stats(p_value, wasserstein_dist, mean, std, coeff, p_value_spearman, test_mean, test_std, gen_mean, gen_std, test_count, gen_count, both_count, len(factor_diff)) # TODO is this correct??

    def _get_ssim(self, image1, image2):
        i1 = denormalize(image1)
        i2 = denormalize(image2)

        return 1 - self.raw_ssim(i1, i2).item()  # 0 means same image, not 1!

    def _load_train_ssim_distances(self):
        df = pd.read_csv(self.path_to_train_ssim_distances, sep=';')

        # order columns alphabetically
        df.reindex(sorted(df.columns), axis=1)
        # order by column 'File'
        df = df.sort_values(by=['File'])
        # convert n x (n+1) Matrix to nxn by removing the 'File' column
        df_square = df.drop('File', axis=1)
        df_square = df_square.drop('Unnamed: 0', axis=1)

        assert len(df_square) == len(df_square.columns)

        return df_square

    def _fit_clustering(self, train_ssim_distances):
        clf = LocalOutlierFactor(n_neighbors=20, metric='precomputed', novelty=True)  # 20 is default value in scikit-learn
        clf.fit(train_ssim_distances)
        return clf

    def _get_images(self):
        generated_images = []
        original_images = []
        original_image_by_filename = {}
        filenames = []
        filename_set = set()

        alter = 'ALTER_'
        counter = 'COUNTER_'

        for image, label, filename in self.generated_test_loader:
            filename_parts = filename[0].split(sep)
            actual_filename = filename_parts[-1]

            if alter in actual_filename:
                actual_filename = actual_filename[6:]
            elif counter in actual_filename:
                actual_filename = actual_filename[8:]
            else:
                raise ValueError('Image is neither alter- nor counterfactual')

            # generated images end in xxx_generated.png, but the originals of course do not --> remove that suffix
            parts = actual_filename.split('.')
            file_extension = parts[1]
            name = parts[0]
            p = name.split('_')
            original_name = '_'.join(p[:-1])
            actual_filename = '.'.join([original_name, file_extension])

            filenames.append(actual_filename)
            filename_set.add(actual_filename)
            generated_images.append(image.cpu().detach())

        for image, label, filename in self.original_test_loader:
            filename_parts = filename[0].split(sep)
            actual_filename = filename_parts[-1]

            if not actual_filename in filename_set:
                continue

            original_image_by_filename[actual_filename] = image.cpu().detach()

        assert len(original_image_by_filename.keys()) == len(filename_set)
        assert len(filename_set) == len(filenames)

        for filename in filenames:
            original_images.append(original_image_by_filename[filename])

        return original_images, generated_images, filenames

    def _get_distances(self, test_images, generated_images):
        # shape: (n_test_images, n_train_images)

        ssim_distances_test = []
        ssim_distances_generated = []

        test_idxs = range(len(test_images))  # same length as generated images
        train_keys = sorted(self.train_images_by_filename.keys())

        product = itertools.product(test_idxs, train_keys)

        ssims_test = []
        ssims_generated = []
        previous_test_idx = -1
        for test_idx, train_key in product:
            if previous_test_idx < 0:
                previous_test_idx = test_idx

            test_image = test_images[test_idx]
            generated_image = generated_images[test_idx]

            if test_idx != previous_test_idx:  # new test image
                ssim_distances_test.append(ssims_test)
                ssim_distances_generated.append(ssims_generated)
                ssims_test = []
                ssims_generated = []
                previous_test_idx = test_idx

            train_image = self.train_images_by_filename[train_key]
            ssim_test = self._get_ssim(test_image, train_image)
            ssims_test.append(ssim_test)

            ssim_generated = self._get_ssim(generated_image, train_image)
            ssims_generated.append(ssim_generated)

        # last iteration
        ssim_distances_test.append(ssims_test)
        ssim_distances_generated.append(ssims_generated)

        assert len(test_images) == len(ssim_distances_test)
        assert len(generated_images) == len(ssim_distances_generated)

        return ssim_distances_test, ssim_distances_generated

    def _get_outlier_factors(self, ssim_distances, clf):
        n_lof = clf.decision_function(ssim_distances)
        return n_lof

    def _plot_distribution(self, outlier_factors, name):
        fig, ax = plt.subplots()
        sns.kdeplot(outlier_factors, shade=False, ax=ax)

        try:
            kdeline = ax.lines[0]
            xs = kdeline.get_xdata()
            ys = kdeline.get_ydata()

            mean = np.mean(outlier_factors)
            std = np.std(outlier_factors)
            left_std = mean - std
            right_std = mean + std

            ax.vlines(mean, 0, np.interp(mean, xs, ys), ls=':')
            ax.fill_between(xs, 0, ys, facecolor='blue', alpha=0.2)
            ax.fill_between(xs, 0, ys, where=(left_std <= xs) & (xs <= right_std), interpolate=True, facecolor='blue',
                            alpha=0.2)
        except IndexError as e:
            pass

        plt.title(f'Negative Local Outlier Factors {name}')
        plt.xlabel('Negative Local Outlier Factor')

        plt.savefig(f'{self.path_to_output_folder}{sep}{name}_nlof_distribution.png', bbox_inches="tight")
        plt.close()

        sns.histplot(outlier_factors)

        plt.title(f'Counts Negative Local Outlier Factors {name}')
        plt.xlabel('Negative Local Outlier Factor')

        plt.savefig(f'{self.path_to_output_folder}{sep}{name}_nlof_count.png', bbox_inches="tight")
        plt.close()

    def _statistical_sameness_test(self, test_outlier_factors, generated_outlier_factors):
        _, p_value = stats.wilcoxon(test_outlier_factors, generated_outlier_factors)  # nullhypothesis: from same distribution
        return p_value

    def _wasserstein_distance(self, test_outlier_factors, generated_outlier_factors):
        return stats.wasserstein_distance(test_outlier_factors, generated_outlier_factors)

    def _get_diff(self, test_outlier_factors, generated_outlier_factors):
        return test_outlier_factors - generated_outlier_factors

    def _plot_factor_diff_distribution(self, factor_diff):
        fig, ax = plt.subplots()
        sns.kdeplot(factor_diff, shade=False, ax=ax)

        try:
            kdeline = ax.lines[0]
            xs = kdeline.get_xdata()
            ys = kdeline.get_ydata()

            mean = np.mean(factor_diff)
            std = np.std(factor_diff)
            left_std = mean - std
            right_std = mean + std

            ax.vlines(mean, 0, np.interp(mean, xs, ys), ls=':')
            ax.fill_between(xs, 0, ys, facecolor='blue', alpha=0.2)
            ax.fill_between(xs, 0, ys, where=(left_std <= xs) & (xs <= right_std), interpolate=True, facecolor='blue',
                            alpha=0.2)
        except IndexError as e:
            pass

        plt.title(f'Difference in Negative Local Outlier Factors for Original and Generated')
        plt.xlabel('Difference in Negative Local Outlier Factor')

        plt.savefig(f'{self.path_to_output_folder}{sep}nlof_diff_distribution.png', bbox_inches="tight")
        plt.close()

        sns.histplot(factor_diff)

        plt.title(f'Counts Difference in Negative Local Outlier Factors for Original and Generated')
        plt.xlabel('Difference in Negative Local Outlier Factor')

        plt.savefig(f'{self.path_to_output_folder}{sep}nlof_diff_count.png', bbox_inches="tight")
        plt.close()

    def _mean_std(self, factor_diff):
        mean = np.mean(factor_diff)
        std = np.std(factor_diff)
        return mean, std

    def _store(self, test_outlier_factors, generated_outlier_factors, factor_diff, filenames):
        print(f'Test LOF: {len(test_outlier_factors)}')
        print(f'Generated LOF: {len(generated_outlier_factors)}')
        print(f'Filenames: {len(filenames)}')
        df = pd.DataFrame(
            {
                'Test LOF': test_outlier_factors,
                'Generated LOF': generated_outlier_factors,
                'Test - Generated LOF': factor_diff,
                'Filename': filenames
            }
        )
        df.to_csv(f'{self.path_to_output_folder}{sep}LOFs.csv', sep=';')

    def _LOF_correlation(self, test_outlier_factors, factor_diff):
        coeff, p_value = stats.spearmanr(test_outlier_factors, factor_diff)
        return coeff, p_value

    def _store_stats(self, p_value_dist_test, wasserstein_dist, mean, std, corr_coeff, corr_p_value, test_mean, test_std, gen_mean, gen_std, test_count, gen_count, both_count, test_len):
        lines = [
            'Comparing the Local Outlier Factors for original test set and generated test set.\n',
            '\n',
            f'Test set mean: {test_mean} -- std: {test_std}\n',
            f'Generated set mean: {gen_mean} -- std: {gen_std}\n',
            f'LOF difference (test - generated) mean: {mean} -- std: {std}\n',
            '\n',
            'Comparing the two distributions directly:\n',
            f'Wasserstein distance: {wasserstein_dist}\n',
            f'Nullhypothesis: same distribution: p-value: {p_value_dist_test}\n',
            '\n',
            'Is there a correlation between the LOF of test set and the difference in the score?\n',
            f'Nullhypothesis: Distributions are uncorrelated: p-value: {corr_p_value}\n',
            f'Spearman coefficient ([-1, 1]): {corr_coeff}\n',
            '\n',
            f'Outliers in the test set: {test_count} of {test_len} -- ({(test_count / test_len):.2f})\n',
            f'Outliers in the generated set: {gen_count} of {test_len} -- ({(gen_count / test_len):.2f})\n',
            f'Cases where both the test and the generated image are outliers: {both_count} of {test_len} -- ({(both_count / test_len):.2f})\n'
        ]

        with open(f'{self.path_to_output_folder}/dist_stats.txt', 'w') as file:
            file.writelines(lines)

    def _get_outlier_count(self, test_train_ssim_distances, generated_train_ssim_distances, clf):
        test_scores = clf.predict(test_train_ssim_distances)
        gen_scores = clf.predict(generated_train_ssim_distances)

        test_count = len(list(filter(lambda x: x < 0, test_scores)))
        gen_count = len(list(filter(lambda x: x < 0, gen_scores)))

        combined = np.array(test_scores) + np.array(gen_scores)
        both_outliers = len(list(filter(lambda x: x == -2, combined)))

        return test_count,  gen_count, both_outliers
