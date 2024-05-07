from os.path import sep

from scipy import stats


class SVMDBDistanceDistributionComparer:

    def __init__(self, name_dist_avg_std_1, name_dist_avg_std_2, output_folder, extension):
        self.name_dist_avg_std_1 = name_dist_avg_std_1
        self.name_dist_avg_std_2 = name_dist_avg_std_2

        self.output_folder = output_folder

        self.remark = '\n'
        self.extension = extension

    def compare(self):
        ks_tests, wasserstein_dists = self._calculate_p_values_and_wasserstein_distances()
        self.print(ks_tests, wasserstein_dists)

    def _calculate_p_values_and_wasserstein_distances(self):
        ks_tests = []
        wasserstein_dists = []

        for ndas1, ndas2 in zip(self.name_dist_avg_std_1, self.name_dist_avg_std_2):
            name1, dist1, _, _ = ndas1
            name2, dist2, _, _ = ndas2

            if name1 == 'Test' and name2 == 'Generated' or name1 == 'Generated' and name2 == 'Test':
                _, p_value = stats.wilcoxon(dist1, dist2)  # nullhypothesis: from same distribution
            else:
                _, p_value = stats.ks_2samp(dist1, dist2)  # nullhypothesis: from same distribution
            ks_tests.append((f'{name1}-{name2}', p_value))

            wasserstein_dist = stats.wasserstein_distance(dist1, dist2)
            wasserstein_dists.append((f'{name1}-{name2}', wasserstein_dist))

        return ks_tests, wasserstein_dists

    def print(self, ks_tests, wasserstein_dists):
        lines = [
            'Comparing the SVM DB Distance distributions:\n',
            self.remark,
            '-----------------------\n',
            '\n',
        ]

        lines.extend([
            '\n',
            'Wilc signed rank: null hypothesis: same distribution (<= 0.05: probably not same):\n',
            '-----------------------\n',
            '\n',
        ])

        for name, p_value in ks_tests:
            lines.append(f'{name}: p-value: {p_value}\n')

        lines.extend([
            '\n',
            'Wasserstein distances:\n',
            '-----------------------\n',
            '\n',
        ])

        for name, dist in wasserstein_dists:
            lines.append(f'{name}: {dist}\n')

        with open(f'{self.output_folder}{sep}ks_tests_wasserstein_{self.extension}.txt', 'w') as file:
            file.writelines(lines)