from abc import abstractmethod
from collections import defaultdict
import numpy as np


class DistributionDistanceBaseCalculator:

    def __init__(self, normalized_weights, mean_weight_0, mean_weight_1, mean_weight_diff):
        self.normalized_weights = normalized_weights
        self.mean_weight_0 = mean_weight_0
        self.mean_weight_1 = mean_weight_1
        self.mean_weight_diff = mean_weight_diff

        self.total_distances_by_row = {}
        self.total_distances_by_row_weighted = {}
        self.total_distances_by_row_unweighted_same_calc_as_weighted = {}

        self.distances_by_feature_by_class = defaultdict(dict)
        self.total_distances = []
        self.total_distances_unweighted_same_calc_as_weighted = []

        self.distances_by_feature_by_class_weighted = defaultdict(dict)
        self.total_distances_weighted = []

        self.critical_feature_mismatches = []

    @abstractmethod
    def _insignificant_hurdle_model(self, feature, label):
        pass

    @abstractmethod
    def _get_dist(self, gen_feature_value, test_feature_value, feature, label, hurdle_model, hurdle_model_of_counter_class, own_median=None, counter_class_median=None):
        pass

    def calculate(self, test_feature_value, gen_feature_value, feature, label, hurdle_model, hurdle_model_of_counter_class, own_median=None, counter_class_median=None, instance_id=-1):
        if self._insignificant_hurdle_model(feature, label):
            return

        cdf_gen, cdf_test, dist = self._get_dist(gen_feature_value, test_feature_value, feature, label, hurdle_model, hurdle_model_of_counter_class, own_median, counter_class_median)

        if dist is not None:
            if int(label) not in self.distances_by_feature_by_class[feature].keys():
                self.distances_by_feature_by_class[feature][int(label)] = []

            self.distances_by_feature_by_class[feature][int(label)].append((dist, cdf_test, cdf_gen))

            if instance_id not in self.total_distances_by_row:
                self.total_distances_by_row[instance_id] = 0
            self.total_distances_by_row[instance_id] = self.total_distances_by_row[instance_id] + dist

            self._add_classifier_weights(cdf_gen, cdf_test, dist, feature, instance_id, label)
        else:
            if gen_feature_value != test_feature_value:
                self.critical_feature_mismatches.append((feature, label, test_feature_value, gen_feature_value))

    def _add_classifier_weights(self, cdf_gen, cdf_test, dist, feature, instance_id, label):
        # weight_class_0 = abs(self.normalized_weights[0][feature].item())
        #
        # weight_class_1 = abs(self.normalized_weights[1][feature].item())

        weight_class_0 = self.normalized_weights[0][feature].item()
        weight_class_1 = self.normalized_weights[1][feature].item()

        weight_diff = np.abs(weight_class_0 - weight_class_1)

        unweighted_dist_same_calc_as_weighted = dist * self.mean_weight_diff #* self.mean_weight_0 + dist * self.mean_weight_1
        weighted_dist = dist * weight_diff#* weight_class_0 + dist * weight_class_1

        if int(label) not in self.distances_by_feature_by_class_weighted[feature].keys():
            self.distances_by_feature_by_class_weighted[feature][int(label)] = []
        self.distances_by_feature_by_class_weighted[feature][int(label)].append((weighted_dist, cdf_test, cdf_gen))

        if instance_id not in self.total_distances_by_row_weighted:
            self.total_distances_by_row_weighted[instance_id] = 0
        self.total_distances_by_row_weighted[instance_id] = self.total_distances_by_row_weighted[
                                                                instance_id] + weighted_dist

        if instance_id not in self.total_distances_by_row_unweighted_same_calc_as_weighted:
            self.total_distances_by_row_unweighted_same_calc_as_weighted[instance_id] = 0
        self.total_distances_by_row_unweighted_same_calc_as_weighted[instance_id] = \
            self.total_distances_by_row_unweighted_same_calc_as_weighted[
                instance_id] + unweighted_dist_same_calc_as_weighted

    def combine(self):
        for instance_id in sorted(self.total_distances_by_row.keys()):
            self.total_distances.append(self.total_distances_by_row[instance_id])

        self._combine_classifier_weighted()

    def _combine_classifier_weighted(self):
        for instance_id in sorted(self.total_distances_by_row_weighted.keys()):
            self.total_distances_weighted.append(self.total_distances_by_row_weighted[instance_id])

        for instance_id in sorted(self.total_distances_by_row_unweighted_same_calc_as_weighted.keys()):
            self.total_distances_unweighted_same_calc_as_weighted.append(
                self.total_distances_by_row_unweighted_same_calc_as_weighted[instance_id])

    def get_distances(self):
        return self.distances_by_feature_by_class, self.total_distances, self.distances_by_feature_by_class_weighted, self.total_distances_weighted, self.total_distances_unweighted_same_calc_as_weighted, self.critical_feature_mismatches


class DistributionDistanceUnsureCalculator(DistributionDistanceBaseCalculator):

    def __init__(self, normalized_weights, mean_weight_0, mean_weight_1, best_p_value_by_feature_by_label):
        super().__init__(normalized_weights, mean_weight_0, mean_weight_1)
        self.best_p_value_by_feature_by_label = best_p_value_by_feature_by_label

    def _insignificant_hurdle_model(self, feature, label):
        return False

    def _get_dist(self, gen_feature_value, test_feature_value, feature, label, hurdle_model, hurdle_model_of_counter_class, own_median=None, counter_class_median=None):
        cdf_test = self._get_cdf(hurdle_model, test_feature_value)
        cdf_gen = self._get_cdf(hurdle_model, gen_feature_value)

        if (cdf_test is not None and cdf_gen is not None) and (not np.isnan(cdf_test) and not np.isnan(cdf_gen)):
            dist = abs(cdf_test - cdf_gen)

        else:
            dist = None
        return cdf_gen, cdf_test, dist

    def _get_cdf(self, hurdle_model, feature_value):
        if hurdle_model is None:
            return None

        cdf = hurdle_model.get_cdf(feature_value)
        return np.array(cdf)


class DistributionDistanceSureCalculator(DistributionDistanceUnsureCalculator):

    def __init__(self, normalized_weights, mean_weight_0, mean_weight_1, best_p_value_by_feature_by_label):
        super().__init__(normalized_weights, mean_weight_0, mean_weight_1, best_p_value_by_feature_by_label)

    def _insignificant_hurdle_model(self, feature, label):
        p_value = self.best_p_value_by_feature_by_label[feature][label]
        return p_value <= 0.05


class DistributionDistanceScaleCalculator(DistributionDistanceBaseCalculator):

    def __init__(self, normalized_weights, mean_weight_0, mean_weight_1, mean_weight_diff, scale_factor_by_feature_by_label):
        super().__init__(normalized_weights, mean_weight_0, mean_weight_1, mean_weight_diff)
        self.scale_factor_by_feature_by_label = scale_factor_by_feature_by_label

    def _get_dist(self, gen_feature_value, test_feature_value, feature, label, hurdle_model=None, hurdle_model_of_counter_class=None, own_median=None, counter_class_median=None):
        if self.scale_factor_by_feature_by_label[feature][label] == 0:
            dist = None
        else:
            dist = abs(test_feature_value - gen_feature_value) #/ self.scale_factor_by_feature_by_label[feature][label]

        if dist is not None and np.isnan(dist): # if scale factor is 0
            dist = None

        assert dist is None or dist >= 0

        return 0, 0, dist

    def _insignificant_hurdle_model(self, feature, label):
        return False  # this entire approach avoids insignificant hurdle models


class UndirectionalDistributionDistanceUnsureCalculator(DistributionDistanceUnsureCalculator):

    def __init__(self, normalized_weights, mean_weight_0, mean_weight_1, best_p_value_by_feature_by_label):
        super().__init__(normalized_weights, mean_weight_0, mean_weight_1, best_p_value_by_feature_by_label)

    def _get_dist(self, gen_feature_value, test_feature_value, feature, label, hurdle_model, hurdle_model_of_counter_class, own_median=None, counter_class_median=None):
        cdf_test = self._get_cdf(hurdle_model, test_feature_value)
        cdf_gen = self._get_cdf(hurdle_model, gen_feature_value)

        if (cdf_test is not None and cdf_gen is not None) and (not np.isnan(cdf_test) and not np.isnan(cdf_gen)):

            if hurdle_model_of_counter_class is None:
                dist = None
            else:
                own_mean = hurdle_model.get_expected_value()
                counter_class_mean = hurdle_model_of_counter_class.get_expected_value()

                if own_mean < counter_class_mean:
                    if gen_feature_value >= test_feature_value:
                        dist = abs(cdf_test - cdf_gen)
                    else:
                        dist = 0
                elif own_mean > counter_class_mean:
                    if gen_feature_value <= test_feature_value:
                        dist = abs(cdf_test - cdf_gen)
                    else:
                        dist = 0
                else:
                    dist = abs(cdf_test - cdf_gen)

                dist = self._add_similarity_weight(dist, feature)
        else:
            dist = None
        return cdf_gen, cdf_test, dist

    def _add_similarity_weight(self, dist, feature):
        # differences in features, were both activation distributions are close are more critical
        return dist


class UndirectionalDistributionDistanceUnsureWeightedCalculator(UndirectionalDistributionDistanceUnsureCalculator):

    def __init__(self, normalized_weights, mean_weight_0, mean_weight_1, best_p_value_by_feature_by_label,
                 wasserstein_distances):
        super().__init__(normalized_weights, mean_weight_0, mean_weight_1, best_p_value_by_feature_by_label)
        self.wasserstein_distances = wasserstein_distances

    def _add_similarity_weight(self, dist, feature):
        # differences in features, were both activation distributions are close are more critical
        distribution_similarity_weight = self.wasserstein_distances[feature]
        sim_factor = 1 / distribution_similarity_weight if distribution_similarity_weight is not 0 else 10000  # arbitrary large number
        dist = dist * sim_factor
        return dist


class UnidirectionalDistributionDistanceSureCalculator(UndirectionalDistributionDistanceUnsureCalculator):

    def __init__(self, normalized_weights, mean_weight_0, mean_weight_1, best_p_value_by_feature_by_label):
        super().__init__(normalized_weights, mean_weight_0, mean_weight_1, best_p_value_by_feature_by_label)

    def _insignificant_hurdle_model(self, feature, label):
        p_value = self.best_p_value_by_feature_by_label[feature][label]
        return p_value <= 0.05


class UnidirectionalDistributionDistanceSureWeightedCalculator(UndirectionalDistributionDistanceUnsureWeightedCalculator):

    def __init__(self, normalized_weights, mean_weight_0, mean_weight_1, best_p_value_by_feature_by_label, wasserstein_distances):
        super().__init__(normalized_weights, mean_weight_0, mean_weight_1, best_p_value_by_feature_by_label, wasserstein_distances)

    def _insignificant_hurdle_model(self, feature, label):
        p_value = self.best_p_value_by_feature_by_label[feature][label]
        return p_value <= 0.05


class UnidirectionalDistributionDistanceScaleCalculator(DistributionDistanceScaleCalculator):

    def __init__(self, normalized_weights, mean_weight_0, mean_weight_1, mean_weight_diff, scale_factor_by_feature_by_label):
        super().__init__(normalized_weights, mean_weight_0, mean_weight_1, mean_weight_diff, scale_factor_by_feature_by_label)

    def _get_dist(self, gen_feature_value, test_feature_value, feature, label, hurdle_model=None, hurdle_model_of_counter_class=None, own_median=None, counter_class_median=None):
        if own_median is None and counter_class_median is None:
            dist = None
        elif own_median is None or counter_class_median is None:
            dist = 10000 # arbitrary large number
        else:
            scale_factor = self.scale_factor_by_feature_by_label[feature][label]
            if scale_factor == 0:
                dist = None
            else:
                if own_median < counter_class_median:
                    if gen_feature_value >= test_feature_value:
                        dist = abs(test_feature_value - gen_feature_value) #/ scale_factor
                    else:
                        dist = 0
                elif own_median > counter_class_median:
                    if gen_feature_value <= test_feature_value:
                        dist = abs(test_feature_value - gen_feature_value) #/ scale_factor
                    else:
                        dist = 0
                else:
                    dist = abs(test_feature_value - gen_feature_value) #/ scale_factor

        # differences in features, were both activation distributions are close are more critical
        dist = self._add_similarity_weight(dist, feature)

        if dist is not None and np.isnan(dist): # if scale factor is 0
            dist = None

        assert dist is None or dist >= 0

        return 0, 0, dist

    def _add_similarity_weight(self, dist, feature):
        return dist


class UnidirectionalDistributionDistanceScaleWeightedCalculator(UnidirectionalDistributionDistanceScaleCalculator):

    def __init__(self, normalized_weights, mean_weight_0, mean_weight_1, scale_factor_by_feature_by_label, wasserstein_distances):
        super().__init__(normalized_weights, mean_weight_0, mean_weight_1, scale_factor_by_feature_by_label)
        self.wasserstein_distances = wasserstein_distances

    def _add_similarity_weight(self, dist, feature):
        # if dist is None:
        #     return dist
        #
        # distribution_similarity_weight = self.wasserstein_distances[feature]
        # sim_factor = 1 / distribution_similarity_weight if distribution_similarity_weight is not 0 else 10000  # arbitrary large number
        # dist = dist * sim_factor
        return dist


class LogitDistributionDistanceUnsureCalculator(DistributionDistanceUnsureCalculator):

    def __init__(self, best_p_value_by_feature_by_label):
        super().__init__(None, None, None, best_p_value_by_feature_by_label)

    def _add_classifier_weights(self, cdf_gen, cdf_test, dist, feature, instance_id, label):
        pass

    def _combine_classifier_weighted(self):
        pass


class LogitDistributionDistanceSureCalculator(DistributionDistanceSureCalculator):

    def __init__(self, best_p_value_by_feature_by_label):
        super().__init__(None, None, None, best_p_value_by_feature_by_label)

    def _add_classifier_weights(self, cdf_gen, cdf_test, dist, feature, instance_id, label):
        pass

    def _combine_classifier_weighted(self):
        pass


class LogitDistributionDistanceScaleCalculator(DistributionDistanceScaleCalculator):

    def __init__(self, scale_factor_by_feature_by_label):
        super().__init__(None, None, None, None, scale_factor_by_feature_by_label)

    def _add_classifier_weights(self, cdf_gen, cdf_test, dist, feature, instance_id, label):
        pass

    def _combine_classifier_weighted(self):
        pass


class LogitUndirectionalDistributionDistanceUnsureCalculator(UndirectionalDistributionDistanceUnsureCalculator):

    def __init__(self, best_p_value_by_feature_by_label):
        super().__init__(None, None, None, best_p_value_by_feature_by_label)

    def _add_classifier_weights(self, cdf_gen, cdf_test, dist, feature, instance_id, label):
        pass

    def _combine_classifier_weighted(self):
        pass


class LogitUndirectionalDistributionDistanceUnsureWeightedCalculator(UndirectionalDistributionDistanceUnsureWeightedCalculator):

    def __init__(self, best_p_value_by_feature_by_label,
                 wasserstein_distances):
        super().__init__(None, None, None, best_p_value_by_feature_by_label,
                         wasserstein_distances)

    def _add_classifier_weights(self, cdf_gen, cdf_test, dist, feature, instance_id, label):
        pass

    def _combine_classifier_weighted(self):
        pass


class LogitUnidirectionalDistributionDistanceSureCalculator(UnidirectionalDistributionDistanceSureCalculator):

    def __init__(self, best_p_value_by_feature_by_label):
        super().__init__(None, None, None, best_p_value_by_feature_by_label)

    def _add_classifier_weights(self, cdf_gen, cdf_test, dist, feature, instance_id, label):
        pass

    def _combine_classifier_weighted(self):
        pass


class LogitUnidirectionalDistributionDistanceSureWeightedCalculator(UnidirectionalDistributionDistanceSureWeightedCalculator):

    def __init__(self, best_p_value_by_feature_by_label,
                 wasserstein_distances):
        super().__init__(None, None, None, best_p_value_by_feature_by_label,
                         wasserstein_distances)

    def _add_classifier_weights(self, cdf_gen, cdf_test, dist, feature, instance_id, label):
        pass

    def _combine_classifier_weighted(self):
        pass


class LogitUnidirectionalDistributionDistanceScaleCalculator(UnidirectionalDistributionDistanceScaleCalculator):

    def __init__(self, scale_factor_by_feature_by_label):
        super().__init__(None, None, None, None, scale_factor_by_feature_by_label)

    def _add_classifier_weights(self, cdf_gen, cdf_test, dist, feature, instance_id, label):
        pass

    def _combine_classifier_weighted(self):
        pass


class LogitUnidirectionalDistributionDistanceScaleWeightedCalculator(UnidirectionalDistributionDistanceScaleWeightedCalculator):

    def __init__(self, scale_factor_by_feature_by_label,
                 wasserstein_distances):
        super().__init__(None, None, None, scale_factor_by_feature_by_label,
                         wasserstein_distances)

    def _add_classifier_weights(self, cdf_gen, cdf_test, dist, feature, instance_id, label):
        pass

    def _combine_classifier_weighted(self):
        pass