from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
from statsmodels.robust import mad
from statsmodels.robust.scale import qn_scale

from main.countercounter.gan._execution.evaluation.activationdistribution.DistributionDistancePartialCalculators import \
    DistributionDistanceUnsureCalculator, DistributionDistanceSureCalculator, DistributionDistanceScaleCalculator, \
    UnidirectionalDistributionDistanceSureCalculator, UndirectionalDistributionDistanceUnsureCalculator, \
    UndirectionalDistributionDistanceUnsureWeightedCalculator, UnidirectionalDistributionDistanceSureWeightedCalculator, \
    UnidirectionalDistributionDistanceScaleCalculator, UnidirectionalDistributionDistanceScaleWeightedCalculator
from main.countercounter.gan._execution.evaluation.activationdistribution.HurdleModel import HurdleModel


class DistributionDistanceCalculator:

    def __init__(self, cnn, best_distribution_models, feature_count, wasserstein_distances, name='feature'):
        self.name = name
        self.cnn = cnn

        self.best_distribution_models = best_distribution_models

        self._get_classifier_weights()

        self.feature_count = feature_count

        self.hurdle_model_stats = []
        self.best_p_value_by_feature_by_label = defaultdict(dict)

        self.hurdle_model_by_class_and_feature = defaultdict(dict)

        self.scale_factor_by_feature_by_label_mad = defaultdict(dict)
        self.scale_factor_by_feature_by_label_qn = defaultdict(dict)

        self.median_by_feature_by_class = defaultdict(dict)

        self.wasserstein_distances = wasserstein_distances

        self._initialize_calculators()

    def _initialize_calculators(self):
        # self.calculator_hurdle_unsure = DistributionDistanceUnsureCalculator(
        #     self.normalized_weights,
        #     self.mean_weight_0,
        #     self.mean_weight_1,
        #     self.best_p_value_by_feature_by_label,
        # )
        # self.calculator_hurdle_significant = DistributionDistanceSureCalculator(
        #     self.normalized_weights,
        #     self.mean_weight_0,
        #     self.mean_weight_1,
        #     self.best_p_value_by_feature_by_label,
        # )
        # self.calculator_mad = DistributionDistanceScaleCalculator(
        #     self.normalized_weights,
        #     self.mean_weight_0,
        #     self.mean_weight_1,
        #     self.scale_factor_by_feature_by_label_mad,
        # )
        self.calculator_qn = DistributionDistanceScaleCalculator(
            self.normalized_weights,
            self.mean_weight_0,
            self.mean_weight_1,
            self.mean_weight_diff,
            self.scale_factor_by_feature_by_label_qn,
        )
        # unidirectional distance calculation
        # self.calculator_hurdle_unsure_uni = UndirectionalDistributionDistanceUnsureCalculator(
        #     self.normalized_weights,
        #     self.mean_weight_0,
        #     self.mean_weight_1,
        #     self.best_p_value_by_feature_by_label,
        # )
        # self.calculator_hurdle_significant_uni = UnidirectionalDistributionDistanceSureCalculator(
        #     self.normalized_weights,
        #     self.mean_weight_0,
        #     self.mean_weight_1,
        #     self.best_p_value_by_feature_by_label,
        # )
        # self.calculator_mad_uni = UnidirectionalDistributionDistanceScaleCalculator(
        #     self.normalized_weights,
        #     self.mean_weight_0,
        #     self.mean_weight_1,
        #     self.scale_factor_by_feature_by_label_mad,
        # )
        self.calculator_qn_uni = UnidirectionalDistributionDistanceScaleCalculator(
            self.normalized_weights,
            self.mean_weight_0,
            self.mean_weight_1,
            self.mean_weight_diff,
            self.scale_factor_by_feature_by_label_qn, # used to be qn, but it is only used to determine whether data is all 0
        )
        # unidirectional distance calculation, weighted by distribution similarity
        # self.calculator_hurdle_unsure_uni_weighted = UndirectionalDistributionDistanceUnsureWeightedCalculator(
        #     self.normalized_weights,
        #     self.mean_weight_0,
        #     self.mean_weight_1,
        #     self.best_p_value_by_feature_by_label,
        #     self.wasserstein_distances,
        # )
        # self.calculator_hurdle_significant_uni_weighted = UnidirectionalDistributionDistanceSureWeightedCalculator(
        #     self.normalized_weights,
        #     self.mean_weight_0,
        #     self.mean_weight_1,
        #     self.best_p_value_by_feature_by_label,
        #     self.wasserstein_distances,
        # )
        # self.calculator_mad_uni_weighted = UnidirectionalDistributionDistanceScaleWeightedCalculator(
        #     self.normalized_weights,
        #     self.mean_weight_0,
        #     self.mean_weight_1,
        #     self.scale_factor_by_feature_by_label_mad,
        #     self.wasserstein_distances,
        # )
        self.calculator_qn_uni_weighted = UnidirectionalDistributionDistanceScaleCalculator(
            self.normalized_weights,
            self.mean_weight_0,
            self.mean_weight_1,
            self.mean_weight_diff,
            self.scale_factor_by_feature_by_label_qn,
            #self.wasserstein_distances,
        )

    def _get_classifier_weights(self):
        self.classifier_weights = self.cnn.classifier.weight.cpu().detach()
        self.normalized_weights = F.normalize(self.classifier_weights, dim=1)

        self.mean_weight_0 = torch.mean(torch.abs(self.normalized_weights[0])).item()
        self.mean_weight_1 = torch.mean(torch.abs(self.normalized_weights[1])).item()

        all_weight_diffs = torch.abs(self.normalized_weights[0] - self.normalized_weights[1])
        self.mean_weight_diff = torch.mean(all_weight_diffs).item()

    def calculate(self, feature_values_test, feature_values_gen, feature, label): # list ordered by instance, just one class
        # idx is number of row
        for idx, (test_feature_value, gen_feature_value) in enumerate(zip(feature_values_test, feature_values_gen)):

            # hurdle_model = self._get_hurdle_model(label, feature)
            # hurdle_model_of_counter_class = self._get_hurdle_model(1 - label, feature)

            own_median = self.median_by_feature_by_class[feature][label]
            counter_class_median = self.median_by_feature_by_class[feature][1 - label]

            # self.calculator_hurdle_unsure.calculate(test_feature_value, gen_feature_value, feature, label, hurdle_model, hurdle_model_of_counter_class, instance_id=idx)
            # self.calculator_hurdle_significant.calculate(test_feature_value, gen_feature_value, feature, label, hurdle_model, hurdle_model_of_counter_class, instance_id=idx)
            # self.calculator_mad.calculate(test_feature_value, gen_feature_value, feature, label, None, None, instance_id=idx)
            self.calculator_qn.calculate(test_feature_value, gen_feature_value, feature, label, None, None, instance_id=idx)

            # self.calculator_hurdle_unsure_uni.calculate(test_feature_value, gen_feature_value, feature, label, hurdle_model, hurdle_model_of_counter_class, instance_id=idx)
            # self.calculator_hurdle_significant_uni.calculate(test_feature_value, gen_feature_value, feature, label, hurdle_model, hurdle_model_of_counter_class, instance_id=idx)
            # self.calculator_mad_uni.calculate(test_feature_value, gen_feature_value, feature, label, None, None, instance_id=idx)
            self.calculator_qn_uni.calculate(test_feature_value, gen_feature_value, feature, label, None, None, instance_id=idx)

            # self.calculator_hurdle_unsure_uni_weighted.calculate(test_feature_value, gen_feature_value, feature, label, hurdle_model, hurdle_model_of_counter_class, instance_id=idx)
            # self.calculator_hurdle_significant_uni_weighted.calculate(test_feature_value, gen_feature_value, feature, label, hurdle_model, hurdle_model_of_counter_class, instance_id=idx)
            # self.calculator_mad_uni_weighted.calculate(test_feature_value, gen_feature_value, feature, label, None, None, own_median, counter_class_median, instance_id=idx)
            self.calculator_qn_uni_weighted.calculate(test_feature_value, gen_feature_value, feature, label, None, None, own_median, counter_class_median, instance_id=idx)

    def combine(self):
        # self.calculator_hurdle_unsure.combine()
        # self.calculator_hurdle_significant.combine()
        # self.calculator_mad.combine()
        self.calculator_qn.combine()

        # self.calculator_hurdle_unsure_uni.combine()
        # self.calculator_hurdle_significant_uni.combine()
        # self.calculator_mad_uni.combine()
        self.calculator_qn_uni.combine()

        # self.calculator_hurdle_unsure_uni_weighted.combine()
        # self.calculator_hurdle_significant_uni_weighted.combine()
        # self.calculator_mad_uni_weighted.combine()
        self.calculator_qn_uni_weighted.combine()

    def get_distances(self):
        # distances_unsure = self.calculator_hurdle_unsure.get_distances()
        # distances_significant = self.calculator_hurdle_significant.get_distances()
        # distances_mad = self.calculator_mad.get_distances()
        distances_qn = self.calculator_qn.get_distances()

        # distances_unsure_uni = self.calculator_hurdle_unsure_uni.get_distances()
        # distances_significant_uni = self.calculator_hurdle_significant_uni.get_distances()
        # distances_mad_uni = self.calculator_mad_uni.get_distances()
        distances_qn_uni = self.calculator_qn_uni.get_distances()

        # distances_unsure_uni_weighted = self.calculator_hurdle_unsure_uni_weighted.get_distances()
        # distances_significant_uni_weighted = self.calculator_hurdle_significant_uni_weighted.get_distances()
        # distances_mad_uni_weighted = self.calculator_mad_uni_weighted.get_distances()
        distances_qn_uni_weighted = self.calculator_qn_uni_weighted.get_distances()

        return None, None, None, distances_qn, \
               None, None, None, distances_qn_uni_weighted, \
               None, None, None, distances_qn_uni

    def prepare(self, feature, train_class_0, train_class_1) -> None:
        # self._make_hurdle_model(feature, train_class_0, train_class_1)
        self._make_scale_factors(feature, train_class_0, train_class_1)
        self._make_median(feature, train_class_0, train_class_1)

    def _make_hurdle_model(self, feature, train_class_0, train_class_1):
        dist0, dist0_p_value, p_value0 = self._add_hurdle_model(train_class_0, 0, feature)
        dist1, dist1_p_value, p_value1 = self._add_hurdle_model(train_class_1, 1, feature)

        self.hurdle_model_stats.append((f'Feature {feature}', dist0, dist0_p_value, p_value0, dist1, dist1_p_value, p_value1))

        self.best_p_value_by_feature_by_label[feature][0] = dist0_p_value
        self.best_p_value_by_feature_by_label[feature][1] = dist1_p_value

    def _get_hurdle_model(self, label, feature_idx):
        return self.hurdle_model_by_class_and_feature[label][feature_idx]

    def _add_hurdle_model(self, train_data, label, feature):
        model = self._get_best_model(train_data, feature, label)
        self.hurdle_model_by_class_and_feature[label][feature] = model

        if model is None:
            return 'all values zero', -1, {}

        dist, dist_p_value, p_value = model.get_stats()
        return dist, dist_p_value, p_value

    def _get_best_model(self, data, feature, label):
        model_rows_feature = self.best_distribution_models[self.best_distribution_models[self.name] == feature]

        if model_rows_feature.empty:
            return None

        model_row = model_rows_feature[model_rows_feature['label'] == label]

        if model_row.empty:
            return None

        dist_name = model_row['distname'].values[0]
        location = model_row['location'].values[0]
        p_value = model_row['pvalue'].values[0]

        dist = dist_name + " " + location
        data = np.array(data)
        model = HurdleModel(data, dist, p_value)

        return model

    def get_hurdle_model_stats(self):
        return self.hurdle_model_stats

    def _make_scale_factors(self, feature, train_class_0, train_class_1):
        scale_factor_0_mad = mad(train_class_0)
        scale_factor_1_mad = mad(train_class_1)

        self.scale_factor_by_feature_by_label_mad[feature][0] = scale_factor_0_mad
        self.scale_factor_by_feature_by_label_mad[feature][1] = scale_factor_1_mad

        scale_factor_0_qn = qn_scale(train_class_0)
        scale_factor_1_qn = qn_scale(train_class_1)

        self.scale_factor_by_feature_by_label_qn[feature][0] = scale_factor_0_qn
        self.scale_factor_by_feature_by_label_qn[feature][1] = scale_factor_1_qn

    def _make_median(self, feature, train_class_0, train_class_1):
        self.median_by_feature_by_class[feature][0] = np.median(train_class_0)
        self.median_by_feature_by_class[feature][1] = np.median(train_class_1)
