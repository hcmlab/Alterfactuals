import numpy as np


class SVMDBDistanceCalculator:

    def __init__(self, svm, datasets, labels):
        self.svm = svm
        self.relevant_columns = sorted([col for col in datasets[0] if col.startswith('feature')])
        self.datasets, self.labels = self._split_by_classes(datasets, labels)

        self.w_norm = np.linalg.norm(self.svm.coef_)

    def calculate(self):
        name_distances_avg_std = []

        for dataset, name in zip(self.datasets, self.labels):
            distances = self._get_distances(dataset)
            avg, std = self._get_avg_std(distances)

            name_distances_avg_std.append((name, distances, avg, std))

        return name_distances_avg_std

    def calculate_signed(self):
        name_distances_avg_std = []

        for dataset, name in zip(self.datasets, self.labels):
            distances = self._get_distances_signed(dataset)
            avg, std = self._get_avg_std(distances)

            name_distances_avg_std.append((name, distances, avg, std))

        return name_distances_avg_std

    def _get_training_data(self, df):
        X_columns = sorted([col for col in df if col.startswith('feature')])
        y_column = 'Class'

        return df[X_columns], df[y_column]

    def _split_by_classes(self, datasets, labels):
        new_sets = []
        new_labels = []

        for dataset, label in zip(datasets, labels):
            new_sets.append(dataset[self.relevant_columns])
            new_sets.append(dataset[dataset['Class'] == 0][self.relevant_columns])
            new_sets.append(dataset[dataset['Class'] == 1][self.relevant_columns])

            new_labels.append(f'{label} -- all classes')
            new_labels.append(f'{label} -- class 0')
            new_labels.append(f'{label} -- class 1')

        return new_sets, new_labels

    def _get_distances(self, dataset):
        return np.abs(self._get_distances_signed(dataset))

    def _get_distances_signed(self, dataset):
        y_proportional_to_distances = self.svm.decision_function(dataset)
        distances = y_proportional_to_distances / self.w_norm

        return distances

    def _get_avg_std(self, distances):
        return np.mean(distances), np.std(distances)