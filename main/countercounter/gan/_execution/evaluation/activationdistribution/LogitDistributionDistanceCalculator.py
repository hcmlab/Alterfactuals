from collections import defaultdict
from typing import Dict
import numpy as np
import torch
import torch.nn.functional as F

from main.countercounter.gan._execution.evaluation.activationdistribution.DistributionDistanceCalculator import \
    DistributionDistanceCalculator
from main.countercounter.gan._execution.evaluation.activationdistribution.DistributionDistancePartialCalculators import \
    LogitDistributionDistanceUnsureCalculator, LogitDistributionDistanceSureCalculator, \
    LogitDistributionDistanceScaleCalculator, LogitUndirectionalDistributionDistanceUnsureCalculator, \
    LogitUnidirectionalDistributionDistanceSureCalculator, LogitUnidirectionalDistributionDistanceScaleCalculator, \
    LogitUndirectionalDistributionDistanceUnsureWeightedCalculator, \
    LogitUnidirectionalDistributionDistanceSureWeightedCalculator, \
    LogitUnidirectionalDistributionDistanceScaleWeightedCalculator
from main.countercounter.gan._execution.evaluation.activationdistribution.HurdleModel import HurdleModel


class LogitDistributionDistanceCalculator(DistributionDistanceCalculator):

    def __init__(self, best_distribution_models, logit_count, wasserstein_distances):
        super().__init__(None, best_distribution_models, logit_count, wasserstein_distances, name='Logit')

    def _get_classifier_weights(self):
        pass

    def _initialize_calculators(self):
        # self.calculator_hurdle_unsure = LogitDistributionDistanceUnsureCalculator(
        #     self.best_p_value_by_feature_by_label,
        # )
        # self.calculator_hurdle_significant = LogitDistributionDistanceSureCalculator(
        #     self.best_p_value_by_feature_by_label,
        # )
        # self.calculator_mad = LogitDistributionDistanceScaleCalculator(
        #     self.scale_factor_by_feature_by_label_mad,
        # )
        self.calculator_qn = LogitDistributionDistanceScaleCalculator(
            self.scale_factor_by_feature_by_label_qn,
        )
        # unidirectional distance calculation
        # self.calculator_hurdle_unsure_uni = LogitUndirectionalDistributionDistanceUnsureCalculator(
        #     self.best_p_value_by_feature_by_label,
        # )
        # self.calculator_hurdle_significant_uni = LogitUnidirectionalDistributionDistanceSureCalculator(
        #     self.best_p_value_by_feature_by_label,
        # )
        # self.calculator_mad_uni = LogitUnidirectionalDistributionDistanceScaleCalculator(
        #     self.scale_factor_by_feature_by_label_mad,
        # )
        self.calculator_qn_uni = LogitUnidirectionalDistributionDistanceScaleCalculator(
            self.scale_factor_by_feature_by_label_qn,
        )
        # unidirectional distance calculation, weighted by distribution similarity
        # self.calculator_hurdle_unsure_uni_weighted = LogitUndirectionalDistributionDistanceUnsureWeightedCalculator(
        #     self.best_p_value_by_feature_by_label,
        #     self.wasserstein_distances,
        # )
        # self.calculator_hurdle_significant_uni_weighted = LogitUnidirectionalDistributionDistanceSureWeightedCalculator(
        #     self.best_p_value_by_feature_by_label,
        #     self.wasserstein_distances,
        # )
        # self.calculator_mad_uni_weighted = LogitUnidirectionalDistributionDistanceScaleWeightedCalculator(
        #     self.scale_factor_by_feature_by_label_mad,
        #     self.wasserstein_distances,
        # )
        self.calculator_qn_uni_weighted = LogitUnidirectionalDistributionDistanceScaleCalculator(
            self.scale_factor_by_feature_by_label_qn,
            #self.wasserstein_distances,
        )
