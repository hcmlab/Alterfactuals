from collections import defaultdict
import numpy as np

from scipy.stats import wasserstein_distance


class WassersteinDistanceCalculator:

    def __init__(self):
        self.wasserstein_distance_by_feature_by_class = {}
        self.wasserstein_distance_by_feature = {}
        self.mean_wasserstein_distance_by_class = {}
        self.mean_wasserstein_distance = None
        self.std_wasserstein_distance = None

        self.distances_between_classes_within_datasets = {}  # shape: feature: (dist_test, dist_generated)
        self.distances_between_classes_between_datasets = {}  # shape: feature: orig0-orig1, alter0-orig1, alter1-orig0

        self.distances_within_train_set = {}

    def get_distances(self):
        return self.wasserstein_distance_by_feature_by_class, self.wasserstein_distance_by_feature, self.mean_wasserstein_distance_by_class, self.mean_wasserstein_distance, self.std_wasserstein_distance

    def calculate_by_feature(self, feature, test_feature_values, gen_feature_values):
        distance = wasserstein_distance(test_feature_values, gen_feature_values)
        self.wasserstein_distance_by_feature[feature] = distance

    def calculate_mean_wasserstein_distance_by_class(self, feature_triviality_by_feature_by_class_test, feature_triviality_by_feature_by_class_gen):
        distances_by_label = defaultdict(list)

        for feature, distances_by_class in self.wasserstein_distance_by_feature_by_class.items():

            for label, distance in distances_by_class.items():
                if self._is_trivial(feature, feature_triviality_by_feature_by_class_gen,
                                    feature_triviality_by_feature_by_class_test, label):
                    continue

                distances_by_label[label].append(distance)

        for label, distances in distances_by_label.items():
            self.mean_wasserstein_distance_by_class[label] = np.mean(distances)

    def _is_trivial(self, feature, feature_triviality_by_feature_by_class_gen,
                    feature_triviality_by_feature_by_class_test, label):
        return feature_triviality_by_feature_by_class_test[feature][label] and \
               feature_triviality_by_feature_by_class_gen[feature][label]

    def calculate_by_feature_by_class(self, feature, label, test_feature_values, gen_feature_values):
        if feature not in self.wasserstein_distance_by_feature_by_class:
            self.wasserstein_distance_by_feature_by_class[feature] = {}

        distance = wasserstein_distance(test_feature_values, gen_feature_values)
        self.wasserstein_distance_by_feature_by_class[feature][label] = distance

    def calculate_mean_wasserstein_distance(self, feature_triviality_by_feature_by_class_test, feature_triviality_by_feature_by_class_gen):
        distances = []

        for feature, distances_by_class in self.wasserstein_distance_by_feature_by_class.items():
            for label, distance in distances_by_class.items():
                if self._is_trivial(feature, feature_triviality_by_feature_by_class_gen,
                                    feature_triviality_by_feature_by_class_test, label):
                    continue

                distances.append(distance)

        self.mean_wasserstein_distance = np.mean(distances)
        self.std_wasserstein_distance = np.std(distances)

    def calculate_distances_between_classes_within_datasets(
            self,
            feature,
            class_0_feature_values_test,
            class_1_feature_values_test,
            class_0_feature_values_gen,
            class_1_feature_values_gen,
    ):
        # idea: calculate distance between class 0 and class 1 on train set and on generated set respectively for each feature

        test_dist = wasserstein_distance(class_0_feature_values_test, class_1_feature_values_test)
        gen_dist = wasserstein_distance(class_0_feature_values_gen, class_1_feature_values_gen)

        self.distances_between_classes_within_datasets[feature] = (test_dist, gen_dist)

    def get_distances_between_classes_within_datasets(self):
        return self.distances_between_classes_within_datasets

    def get_distance_between_classes_between_datasets(self):
        return self.distances_between_classes_between_datasets

    def calculate_distance_between_classes_between_datasets(
            self,
            feature,
            test_0_feature,
            test_1_feature,
            gen_0_feature,
            gen_1_feature,
    ):
        test0_test1 = wasserstein_distance(test_0_feature, test_1_feature)
        gen0_test1 = wasserstein_distance(gen_0_feature, test_1_feature)
        gen1_test0 = wasserstein_distance(gen_1_feature, test_0_feature)

        self.distances_between_classes_between_datasets[feature] = (test0_test1, gen0_test1, gen1_test0)

    def calculate_distances_between_classes_within_dataset(
            self,
            feature,
            train_class_0,
            train_class_1,
    ):
        # shape: feature: dist_train

        train_dist = wasserstein_distance(train_class_0, train_class_1)
        self.distances_within_train_set[feature] = train_dist
