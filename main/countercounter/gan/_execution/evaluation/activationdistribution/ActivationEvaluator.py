from collections import defaultdict
from os.path import sep

from joblib import load

import torch
import pandas as pd
import os

from main.countercounter.classifier.emotion_classifier.CustomNets import SoftmaxLogitWrapper
from main.countercounter.classifier.evaluation.CombinedDistributionPlotter import CombinedDistributionPlotter
from main.countercounter.classifier.evaluation.DistributionCalculator import DistributionCalculator
from main.countercounter.classifier.evaluation.DistributionPlotter import DistributionPlotter
from main.countercounter.classifier.evaluation.LogitCombinedDistributionPlotter import LogitCombinedDistributionPlotter
from main.countercounter.classifier.evaluation.LogitDistributionPlotter import LogitDistributionPlotter
from main.countercounter.classifier.evaluation.SVMDBDistanceCalculator import SVMDBDistanceCalculator
from main.countercounter.classifier.evaluation.SVMDBDistancePlotter import SVMDBDistancePlotter
from main.countercounter.gan._execution.evaluation.activationdistribution.ActivationDistributionComparer import \
    ActivationDistributionComparer
from main.countercounter.gan._execution.evaluation.activationdistribution.DistributionDistanceCalculator import \
    DistributionDistanceCalculator
from main.countercounter.gan._execution.evaluation.activationdistribution.DistributionDistancePlotter import \
    DistributionDistancePlotter
from main.countercounter.gan._execution.evaluation.activationdistribution.LogitActivationDistributionComparer import \
    LogitActivationDistributionComparer
from main.countercounter.gan._execution.evaluation.activationdistribution.LogitDistributionDistanceCalculator import \
    LogitDistributionDistanceCalculator
from main.countercounter.gan._execution.evaluation.activationdistribution.LogitWassersteinDistanceCalculator import \
    LogitWassersteinDistanceCalculator
from main.countercounter.gan._execution.evaluation.activationdistribution.SVMDBDistanceComparer import \
    SVMDBDistanceComparer
from main.countercounter.gan._execution.evaluation.activationdistribution.SVMDBDistanceComparisonPlotter import \
    SVMDBDistanceComparisonPlotter
from main.countercounter.gan._execution.evaluation.activationdistribution.SVMDBDistanceDistributionComparer import \
    SVMDBDistanceDistributionComparer
from main.countercounter.gan._execution.evaluation.activationdistribution.UnidirectionalSVMDBDistanceComparer import \
    UnidirectionalSVMDBDistanceComparer
from main.countercounter.gan._execution.evaluation.activationdistribution.WassersteinDistanceCalculator import \
    WassersteinDistanceCalculator
from main.countercounter.gan._execution.evaluation.activationdistribution.WassersteinDistancePlotter import \
    WassersteinDistancePlotter
from main.countercounter.gan._execution.evaluation.activationdistribution.WeightActivationCalculator import \
    WeightActivationCalculator
from main.countercounter.gan._execution.evaluation.activationdistribution.WeightActivationPlotter import \
    WeightActivationPlotter
from main.countercounter.gan.utils.AbstractTraining import DEVICE



class ActivationEvaluator:

    def __init__(
            self,
            setup, path_to_output_folder='',
            path_to_distribution_csv=None,
            path_to_logit_csv=None,
            path_to_distribution_models=None,
            path_to_logit_models=None,
            svm_path=None,
            csv_act_path1=None,
            csv_act_path2=None,
            csv_logit_path_1=None,
            csv_logit_path_2=None,
    ):
        self.test_loader = setup.test_loader

        if not isinstance(setup.classifier.model, SoftmaxLogitWrapper):
            raise NotImplementedError
        self.classifier = setup.classifier

        self.path_to_output_folder = path_to_output_folder
        self.path_to_distribution_csv = path_to_distribution_csv
        self.path_to_logit_csv = path_to_logit_csv

        self.test_data = []
        self.generated_data = []

        self.path_to_distribution_models = path_to_distribution_models
        self.best_distribution_models = self._load_best_distribution_models()
        self.path_to_logit_models = path_to_logit_models
        self._best_logit_distribution_models = self._load_best_logit_distribution_models()
        
        self.svm_path = svm_path

        self.from_csv = csv_act_path1 is not None and csv_act_path2 is not None
        self.csv_act_path1 = csv_act_path1
        self.csv_act_path2 = csv_act_path2
        self.csv_logit_path_1 = csv_logit_path_1
        self.csv_logit_path_2 = csv_logit_path_2

        if not self.from_csv:
            self.generator = setup.generator.to(DEVICE)

        self.df_logit_feature_count = 2

        self.wasserstein_distances_within_train_set = None
        self.logit_wasserstein_distances_within_train_set = None

    def evaluate(self):
        print(f'Evaluating:')
        print('---------------------------------------------------------')

        if self.from_csv:
            print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX LOADINIG FROM CSV')
            df1_path = self.csv_act_path1
            df2_path = self.csv_act_path2

            self.df_test = pd.read_csv(df1_path, sep=';')
            self.df_generated = pd.read_csv(df2_path, sep=';')

            self.df_test_feature_count = self._get_feature_count(self.df_test)
            self.df_generated_feature_count = self._get_feature_count(self.df_generated)

            self.df_logit_test = pd.read_csv(self.csv_logit_path_1, sep=';')
            self.df_logit_generated = pd.read_csv(self.csv_logit_path_2, sep=';')

            self.df_logit_feature_count = 2

        else:
            count = 0
            for imgs, labels in self.test_loader:
                # if count > 10:
                #     break
                count += 1
                imgs = imgs.to(DEVICE)
                generated = self.generator(imgs, labels)

                pred_original = self.classifier.pred(imgs)
                label_original = torch.argmax(pred_original, 1)

                pred_generated = self.classifier.pred(generated)
                labels_generated = torch.argmax(pred_generated, 1)

                if label_original == labels_generated:  # only on actual alterfactuals
                    self.test_data.append((imgs.cpu().detach(), label_original.item()))
                    self.generated_data.append((generated.cpu().detach(), labels_generated.item()))

            self.df_test, self.df_test_feature_count, self.df_logit_test = DistributionCalculator(self.test_data, self.classifier.model.model).calculate()
            self.df_generated, self.df_generated_feature_count, self.df_logit_generated = DistributionCalculator(self.generated_data, self.classifier.model.model).calculate()

        self.abs_weight_by_feature_by_class, self.abs_weight_by_feature, self.weight_by_feature_by_class = WeightActivationCalculator(
            self.df_test_feature_count, self.classifier.model.model).calculate()

        self.df_train, self.df_logit_train = self._load_train_distributions()

        print('Starting eval on act dataframes')
        self._eval_on_act_dataframes()
        print('Starting eval on logit dataframes')
        self._eval_on_logit_dataframes()

    def _eval_on_act_dataframes(self):
        self._plot_db_distances()
        print('SVM distances done')
        self._calculate_activation_evaluation()

    def _calculate_activation_evaluation(self):
        feature_triviality_by_feature_by_class_train = defaultdict(dict)
        feature_triviality_by_feature_by_class_test = defaultdict(dict)
        feature_triviality_by_feature_by_class_gen = defaultdict(dict)

        # all output paths
        path_dist_plot_test, path_dist_plot_gen, path_comparison = self._mk_output_dirs()

        critical_mismatch_path_significant,\
        critical_mismatch_path_unsure,\
        hurdle_model_stats_path,\
        ks_path,\
        no_weights_path_mad,\
        no_weights_path_qn,\
        no_weights_path_significant,\
        no_weights_path_significant_uni_unweighted,\
        no_weights_path_significant_uni_weighted,\
        no_weights_path_uni_unweighted_mad,\
        no_weights_path_uni_unweighted_qn,\
        no_weights_path_uni_weighted_mad,\
        no_weights_path_uni_weighted_qn,\
        no_weights_path_unsure,\
        no_weights_path_unsure_uni_unweighted,\
        no_weights_path_unsure_uni_weighted,\
        wasserstein_path,\
        weights_path_mad,\
        weights_path_qn,\
        weights_path_significant,\
        weights_path_significant_uni_unweighted,\
        weights_path_significant_uni_weighted,\
        weights_path_uni_unweighted_mad,\
        weights_path_uni_unweighted_qn,\
        weights_path_uni_weighted_mad,\
        weights_path_uni_weighted_qn,\
        weights_path_unsure,\
        weights_path_unsure_uni_unweighted,\
        weights_path_unsure_uni_weighted = self._make_activation_paths()

        print('Output paths done')

        # calculators and plotters
        test_dist_plotter = DistributionPlotter(path_dist_plot_test)
        gen_dist_plotter = DistributionPlotter(path_dist_plot_gen)
        combined_dist_plotter = CombinedDistributionPlotter(
            path_comparison,
        )

        wasserstein_dist_calculator = WassersteinDistanceCalculator()
        self.wasserstein_distances_within_train_set = wasserstein_dist_calculator.distances_within_train_set

        act_dist_comparer = ActivationDistributionComparer(ks_path)

        distribution_distance_calculator = DistributionDistanceCalculator(self.classifier.model.model, self.best_distribution_models, self.df_test_feature_count, self.wasserstein_distances_within_train_set)

        print('Starting to iterate over features')
        # data
        for feature in range(self.df_test_feature_count):
            column = f'feature_{feature}'

            self._calculate_activation_on_feature(
                act_dist_comparer,
                column,
                combined_dist_plotter,
                distribution_distance_calculator,
                feature,
                feature_triviality_by_feature_by_class_gen,
                feature_triviality_by_feature_by_class_test,
                feature_triviality_by_feature_by_class_train,
                gen_dist_plotter,
                test_dist_plotter,
                wasserstein_dist_calculator,
            )
            print('Activations on features done')

        self._calculate_activation_over_features(
            act_dist_comparer,
            feature_triviality_by_feature_by_class_gen,
            feature_triviality_by_feature_by_class_test,
            feature_triviality_by_feature_by_class_train,
            wasserstein_dist_calculator,
            wasserstein_path,
        )
        print('Activation over features done')

        self._calculate_pair_distances(
            critical_mismatch_path_significant,
            critical_mismatch_path_unsure,
            distribution_distance_calculator,
            hurdle_model_stats_path,
            no_weights_path_mad,
            no_weights_path_qn,
            no_weights_path_significant,
            no_weights_path_significant_uni_unweighted,
            no_weights_path_significant_uni_weighted,
            no_weights_path_uni_unweighted_mad,
            no_weights_path_uni_unweighted_qn,
            no_weights_path_uni_weighted_mad,
            no_weights_path_uni_weighted_qn,
            no_weights_path_unsure,
            no_weights_path_unsure_uni_unweighted,
            no_weights_path_unsure_uni_weighted,
            weights_path_mad,
            weights_path_qn,
            weights_path_significant,
            weights_path_significant_uni_unweighted,
            weights_path_significant_uni_weighted,
            weights_path_uni_unweighted_mad,
            weights_path_uni_unweighted_qn,
            weights_path_uni_weighted_mad,
            weights_path_uni_weighted_qn,
            weights_path_unsure,
            weights_path_unsure_uni_unweighted,
            weights_path_unsure_uni_weighted,
        )
        print('Calculation of pair distances done')

    def _calculate_pair_distances(
            self,
            critical_mismatch_path_significant,
            critical_mismatch_path_unsure,
            distribution_distance_calculator,
            hurdle_model_stats_path,
            no_weights_path_mad,
            no_weights_path_qn,
            no_weights_path_significant,
            no_weights_path_significant_uni_unweighted,
            no_weights_path_significant_uni_weighted,
            no_weights_path_uni_unweighted_mad,
            no_weights_path_uni_unweighted_qn,
            no_weights_path_uni_weighted_mad,
            no_weights_path_uni_weighted_qn,
            no_weights_path_unsure,
            no_weights_path_unsure_uni_unweighted,
            no_weights_path_unsure_uni_weighted,
            weights_path_mad,
            weights_path_qn,
            weights_path_significant,
            weights_path_significant_uni_unweighted,
            weights_path_significant_uni_weighted,
            weights_path_uni_unweighted_mad,
            weights_path_uni_unweighted_qn,
            weights_path_uni_weighted_mad,
            weights_path_uni_weighted_qn,
            weights_path_unsure,
            weights_path_unsure_uni_unweighted,
            weights_path_unsure_uni_weighted,
                                  ):
        distribution_distance_calculator.combine()

        distances_unsure, \
        distances_significant, \
        distances_mad, \
        distances_qn, \
        distances_unsure_uni_weighted, \
        distances_significant_uni_weighted, \
        distances_uni_weighted_mad, \
        distances_uni_weighted_qn, \
        distances_unsure_uni_unweighted, \
        distances_significant_uni_unweighted, \
        distances_uni_unweighted_mad, \
        distances_uni_unweighted_qn = distribution_distance_calculator.get_distances()

        # all hurdle models

        # distances_by_feature_by_class_unsure, total_distances_unsure, distances_by_feature_by_class_weighted_unsure, total_distances_weighted_unsure, total_distances_unweighted_unsure, critical_feature_mismatches_unsure = distances_unsure
        # hurdle_model_stats = distribution_distance_calculator.get_hurdle_model_stats()
        #
        # DistributionDistancePlotter(distances_by_feature_by_class_unsure, total_distances_unsure, no_weights_path_unsure, total_distances_unweighted_unsure).plot()
        # DistributionDistancePlotter(distances_by_feature_by_class_weighted_unsure, total_distances_weighted_unsure, weights_path_unsure).plot()
        #
        # CriticalFeatureMismatchPlotter(critical_feature_mismatches_unsure, critical_mismatch_path_unsure).plot()
        # HurdleModelStatsPrinter(hurdle_model_stats, hurdle_model_stats_path).print('Features')
        #
        # # only significant hurdle models
        # distances_by_feature_by_class_significant, total_distances_significant, distances_by_feature_by_class_weighted_significant, total_distances_weighted_significant, total_distances_unweighted_significant, critical_feature_mismatches_significant = distances_significant
        #
        # DistributionDistancePlotter(distances_by_feature_by_class_significant, total_distances_significant, no_weights_path_significant, total_distances_unweighted_significant).plot()
        # DistributionDistancePlotter(distances_by_feature_by_class_weighted_significant, total_distances_weighted_significant, weights_path_significant).plot()
        # CriticalFeatureMismatchPlotter(critical_feature_mismatches_significant, critical_mismatch_path_significant).plot()
        #
        # # MAD like Wachter
        # distances_by_feature_by_class_mad, total_distances_mad, distances_by_feature_by_class_weighted_mad, total_distances_weighted_mad, total_distances_unweighted_mad, critical_feature_mismatches_mad = distances_mad
        #
        # DistributionDistancePlotter(distances_by_feature_by_class_mad, total_distances_mad, no_weights_path_mad, total_distances_unweighted_mad).plot()
        # DistributionDistancePlotter(distances_by_feature_by_class_weighted_mad, total_distances_weighted_mad, weights_path_mad).plot()

        # Qn
        distances_by_feature_by_class_qn, total_distances_qn, distances_by_feature_by_class_weighted_qn, total_distances_weighted_qn, total_distances_unweighted_qn, critical_feature_mismatches_qn = distances_qn

        DistributionDistancePlotter(distances_by_feature_by_class_qn, total_distances_qn, no_weights_path_qn, total_distances_unweighted_qn).plot()
        DistributionDistancePlotter(distances_by_feature_by_class_weighted_qn, total_distances_weighted_qn, weights_path_qn).plot()

        # self._plot_weight_vs_activation_diff(distances_by_feature_by_class_unsure, '_all_models')
        # self._plot_weight_vs_activation_diff(distances_by_feature_by_class_significant, '_significant_hurdle_models')
        # self._plot_weight_vs_activation_diff(distances_by_feature_by_class_mad, '_MAD')
        self._plot_weight_vs_activation_diff(distances_by_feature_by_class_qn, '_BASE')

        print('Bidirectional distances done, starting unidirectional')
        # again, but unidirectional
        # distances_by_feature_by_class_unsure_uni_weighted, total_distances_unsure_uni_weighted, distances_by_feature_by_class_weighted_unsure_uni_weighted, total_distances_weighted_unsure_uni_weighted, total_distances_unweighted_unsure_uni_weighted, critical_feature_mismatches_unsure_uni_weighted = distances_unsure_uni_weighted
        #
        # DistributionDistancePlotter(distances_by_feature_by_class_unsure_uni_weighted, total_distances_unsure_uni_weighted, no_weights_path_unsure_uni_weighted, total_distances_unweighted_unsure_uni_weighted).plot()
        # DistributionDistancePlotter(distances_by_feature_by_class_weighted_unsure_uni_weighted, total_distances_weighted_unsure_uni_weighted, weights_path_unsure_uni_weighted).plot()
        #
        # distances_by_feature_by_class_significant_uni_weighted, total_distances_significant_uni_weighted, distances_by_feature_by_class_weighted_significant_uni_weighted, total_distances_weighted_significant_uni_weighted, total_distances_unweighted_significant_uni_weighted, critical_feature_mismatches_significant_uni_weighted = distances_significant_uni_weighted
        #
        # DistributionDistancePlotter(distances_by_feature_by_class_significant_uni_weighted, total_distances_significant_uni_weighted, no_weights_path_significant_uni_weighted, total_distances_unweighted_significant_uni_weighted).plot()
        # DistributionDistancePlotter(distances_by_feature_by_class_weighted_significant_uni_weighted, total_distances_weighted_significant_uni_weighted, weights_path_significant_uni_weighted).plot()
        #
        # distances_by_feature_by_class_uni_weighted_mad, total_distances_uni_weighted_mad, distances_by_feature_by_class_weighted_uni_weighted_mad, total_distances_weighted_uni_weighted_mad, total_distances_unweighted_uni_weighted_mad, critical_feature_mismatches_mad_uni_weighted = distances_uni_weighted_mad
        #
        # DistributionDistancePlotter(distances_by_feature_by_class_uni_weighted_mad, total_distances_uni_weighted_mad, no_weights_path_uni_weighted_mad, total_distances_unweighted_uni_weighted_mad).plot()
        # DistributionDistancePlotter(distances_by_feature_by_class_weighted_uni_weighted_mad, total_distances_weighted_uni_weighted_mad, weights_path_uni_weighted_mad).plot()

        distances_by_feature_by_class_uni_weighted_qn, total_distances_uni_weighted_qn, distances_by_feature_by_class_weighted_uni_weighted_qn, total_distances_weighted_uni_weighted_qn, total_distances_unweighted_uni_weighted_qn, critical_feature_mismatches_qn_uni_weighted = distances_uni_weighted_qn

        DistributionDistancePlotter(distances_by_feature_by_class_uni_weighted_qn, total_distances_uni_weighted_qn, no_weights_path_uni_weighted_qn, total_distances_unweighted_uni_weighted_qn).plot()
        DistributionDistancePlotter(distances_by_feature_by_class_weighted_uni_weighted_qn, total_distances_weighted_uni_weighted_qn, weights_path_uni_weighted_qn).plot()

        # self._plot_weight_vs_activation_diff(distances_by_feature_by_class_unsure_uni_weighted, '_all_models_unidirectional')
        # self._plot_weight_vs_activation_diff(distances_by_feature_by_class_significant_uni_weighted, '_significant_hurdle_models_unidirectional')
        # self._plot_weight_vs_activation_diff(distances_by_feature_by_class_uni_weighted_mad, '_MAD_unidirectional')
        self._plot_weight_vs_activation_diff(distances_by_feature_by_class_uni_weighted_qn, '_BASE_unidirectional')


        print('unidirectional done, starting unidirectional unweighted by dist sim')
        # and again, but only unidirectional activation deviations without wasserstein distance weight for distribution similarity
        # distances_by_feature_by_class_unsure_uni_unweighted, total_distances_unsure_uni_unweighted, distances_by_feature_by_class_weighted_unsure_uni_unweighted, total_distances_weighted_unsure_uni_unweighted, total_distances_unweighted_unsure_uni_unweighted, critical_feature_mismatches_unsure_uni_unweighted = distances_unsure_uni_unweighted
        #
        # DistributionDistancePlotter(distances_by_feature_by_class_unsure_uni_unweighted, total_distances_unsure_uni_unweighted, no_weights_path_unsure_uni_unweighted, total_distances_unweighted_unsure_uni_unweighted).plot()
        # DistributionDistancePlotter(distances_by_feature_by_class_weighted_unsure_uni_unweighted, total_distances_weighted_unsure_uni_unweighted, weights_path_unsure_uni_unweighted).plot()
        #
        # distances_by_feature_by_class_significant_uni_unweighted, total_distances_significant_uni_unweighted, distances_by_feature_by_class_weighted_significant_uni_unweighted, total_distances_weighted_significant_uni_unweighted, total_distances_unweighted_significant_uni_unweighted, critical_feature_mismatches_significant_uni_unweighted = distances_significant_uni_unweighted
        #
        # DistributionDistancePlotter(distances_by_feature_by_class_significant_uni_unweighted, total_distances_significant_uni_unweighted, no_weights_path_significant_uni_unweighted, total_distances_unweighted_significant_uni_unweighted).plot()
        # DistributionDistancePlotter(distances_by_feature_by_class_weighted_significant_uni_unweighted, total_distances_weighted_significant_uni_unweighted, weights_path_significant_uni_unweighted).plot()
        #
        # distances_by_feature_by_class_uni_unweighted_mad, total_distances_uni_unweighted_mad, distances_by_feature_by_class_weighted_uni_unweighted_mad, total_distances_weighted_uni_unweighted_mad, total_distances_unweighted_uni_unweighted_mad, critical_feature_mismatches_mad_uni_unweighted = distances_uni_unweighted_mad
        #
        # DistributionDistancePlotter(distances_by_feature_by_class_uni_unweighted_mad, total_distances_uni_unweighted_mad, no_weights_path_uni_unweighted_mad, total_distances_unweighted_uni_unweighted_mad).plot()
        # DistributionDistancePlotter(distances_by_feature_by_class_weighted_uni_unweighted_mad, total_distances_weighted_uni_unweighted_mad, weights_path_uni_unweighted_mad).plot()

        distances_by_feature_by_class_uni_unweighted_qn, total_distances_uni_unweighted_qn, distances_by_feature_by_class_weighted_uni_unweighted_qn, total_distances_weighted_uni_unweighted_qn, total_distances_unweighted_uni_unweighted_qn, critical_feature_mismatches_qn_uni_unweighted = distances_uni_unweighted_qn

        DistributionDistancePlotter(distances_by_feature_by_class_uni_unweighted_qn, total_distances_uni_unweighted_qn, no_weights_path_uni_unweighted_qn, total_distances_unweighted_uni_unweighted_qn).plot()
        DistributionDistancePlotter(distances_by_feature_by_class_weighted_uni_unweighted_qn, total_distances_weighted_uni_unweighted_qn, weights_path_uni_unweighted_qn).plot()

        # self._plot_weight_vs_activation_diff(distances_by_feature_by_class_unsure_uni_unweighted, '_all_models_unidirectional_unweighted_by_dist_sim')
        # self._plot_weight_vs_activation_diff(distances_by_feature_by_class_significant_uni_unweighted, '_significant_hurdle_models_unidirectional_unweighted_by_dist_sim')
        # self._plot_weight_vs_activation_diff(distances_by_feature_by_class_uni_unweighted_mad, '_MAD_unidirectional_unweighted_by_dist_sim')
        self._plot_weight_vs_activation_diff(distances_by_feature_by_class_uni_unweighted_qn, '_BASE_unidirectional_unweighted_by_dist_sim')

    def _calculate_pair_logit_distances(
            self,
            critical_mismatch_path_significant,
            critical_mismatch_path_unsure,
            distribution_distance_calculator,
            hurdle_model_stats_path,
            no_weights_path_mad,
            no_weights_path_qn,
            no_weights_path_significant,
            no_weights_path_significant_uni_unweighted,
            no_weights_path_significant_uni_weighted,
            no_weights_path_uni_unweighted_mad,
            no_weights_path_uni_unweighted_qn,
            no_weights_path_uni_weighted_mad,
            no_weights_path_uni_weighted_qn,
            no_weights_path_unsure,
            no_weights_path_unsure_uni_unweighted,
            no_weights_path_unsure_uni_weighted,
    ):
        # logits are not weighted by cnn weights

        distribution_distance_calculator.combine()

        distances_unsure, \
        distances_significant, \
        distances_mad, \
        distances_qn, \
        distances_unsure_uni_weighted, \
        distances_significant_uni_weighted, \
        distances_uni_weighted_mad, \
        distances_uni_weighted_qn, \
        distances_unsure_uni_unweighted, \
        distances_significant_uni_unweighted, \
        distances_uni_unweighted_mad, \
        distances_uni_unweighted_qn = distribution_distance_calculator.get_distances()

        # all hurdle models

        # distances_by_feature_by_class_unsure, total_distances_unsure, distances_by_feature_by_class_weighted_unsure, total_distances_weighted_unsure, total_distances_unweighted_unsure, critical_feature_mismatches_unsure = distances_unsure
        # hurdle_model_stats = distribution_distance_calculator.get_hurdle_model_stats()
        #
        # DistributionDistancePlotter(distances_by_feature_by_class_unsure, total_distances_unsure,
        #                             no_weights_path_unsure, total_distances_unweighted_unsure).plot()
        #
        # CriticalFeatureMismatchPlotter(critical_feature_mismatches_unsure, critical_mismatch_path_unsure).plot()
        # HurdleModelStatsPrinter(hurdle_model_stats, hurdle_model_stats_path).print('Features')
        #
        # # only significant hurdle models
        # distances_by_feature_by_class_significant, total_distances_significant, distances_by_feature_by_class_weighted_significant, total_distances_weighted_significant, total_distances_unweighted_significant, critical_feature_mismatches_significant = distances_significant
        #
        # DistributionDistancePlotter(distances_by_feature_by_class_significant, total_distances_significant,
        #                             no_weights_path_significant, total_distances_unweighted_significant).plot()
        #
        # CriticalFeatureMismatchPlotter(critical_feature_mismatches_significant,
        #                                critical_mismatch_path_significant).plot()
        #
        # # MAD like Wachter
        # distances_by_feature_by_class_mad, total_distances_mad, distances_by_feature_by_class_weighted_mad, total_distances_weighted_mad, total_distances_unweighted_mad, critical_feature_mismatches_mad = distances_mad
        #
        # DistributionDistancePlotter(distances_by_feature_by_class_mad, total_distances_mad, no_weights_path_mad,
        #                             total_distances_unweighted_mad).plot()

        # Qn
        distances_by_feature_by_class_qn, total_distances_qn, distances_by_feature_by_class_weighted_qn, total_distances_weighted_qn, total_distances_unweighted_qn, critical_feature_mismatches_qn = distances_qn

        DistributionDistancePlotter(distances_by_feature_by_class_qn, total_distances_qn, no_weights_path_qn,
                                    total_distances_unweighted_qn).plot()

        # self._plot_weight_vs_activation_diff(distances_by_feature_by_class_unsure, '_all_models_logit')
        # self._plot_weight_vs_activation_diff(distances_by_feature_by_class_significant,
        #                                      '_significant_hurdle_models_logit')
        # self._plot_weight_vs_activation_diff(distances_by_feature_by_class_mad, '_MAD_logit')
        self._plot_weight_vs_activation_diff(distances_by_feature_by_class_qn, '_BASE_logit')

        print('bidirectional logit done')
        # again, but unidirectional
        # distances_by_feature_by_class_unsure_uni_weighted, total_distances_unsure_uni_weighted, distances_by_feature_by_class_weighted_unsure_uni_weighted, total_distances_weighted_unsure_uni_weighted, total_distances_unweighted_unsure_uni_weighted, critical_feature_mismatches_unsure_uni_weighted = distances_unsure_uni_weighted
        #
        # DistributionDistancePlotter(distances_by_feature_by_class_unsure_uni_weighted,
        #                             total_distances_unsure_uni_weighted, no_weights_path_unsure_uni_weighted,
        #                             total_distances_unweighted_unsure_uni_weighted).plot()
        #
        # distances_by_feature_by_class_significant_uni_weighted, total_distances_significant_uni_weighted, distances_by_feature_by_class_weighted_significant_uni_weighted, total_distances_weighted_significant_uni_weighted, total_distances_unweighted_significant_uni_weighted, critical_feature_mismatches_significant_uni_weighted = distances_significant_uni_weighted
        #
        # DistributionDistancePlotter(distances_by_feature_by_class_significant_uni_weighted,
        #                             total_distances_significant_uni_weighted,
        #                             no_weights_path_significant_uni_weighted,
        #                             total_distances_unweighted_significant_uni_weighted).plot()
        #
        # distances_by_feature_by_class_uni_weighted_mad, total_distances_uni_weighted_mad, distances_by_feature_by_class_weighted_uni_weighted_mad, total_distances_weighted_uni_weighted_mad, total_distances_unweighted_uni_weighted_mad, critical_feature_mismatches_mad_uni_weighted = distances_uni_weighted_mad
        #
        # DistributionDistancePlotter(distances_by_feature_by_class_uni_weighted_mad,
        #                             total_distances_uni_weighted_mad, no_weights_path_uni_weighted_mad,
        #                             total_distances_unweighted_uni_weighted_mad).plot()

        distances_by_feature_by_class_uni_weighted_qn, total_distances_uni_weighted_qn, distances_by_feature_by_class_weighted_uni_weighted_qn, total_distances_weighted_uni_weighted_qn, total_distances_unweighted_uni_weighted_qn, critical_feature_mismatches_qn_uni_weighted = distances_uni_weighted_qn

        DistributionDistancePlotter(distances_by_feature_by_class_uni_weighted_qn, total_distances_uni_weighted_qn,
                                    no_weights_path_uni_weighted_qn,
                                    total_distances_unweighted_uni_weighted_qn).plot()

        # self._plot_weight_vs_activation_diff(distances_by_feature_by_class_unsure_uni_weighted,
        #                                      '_all_models_unidirectional_logit')
        # self._plot_weight_vs_activation_diff(distances_by_feature_by_class_significant_uni_weighted,
        #                                      '_significant_hurdle_models_unidirectional_logit')
        # self._plot_weight_vs_activation_diff(distances_by_feature_by_class_uni_weighted_mad, '_MAD_unidirectional_logit')
        self._plot_weight_vs_activation_diff(distances_by_feature_by_class_uni_weighted_qn, '_BASE_unidirectional_logit')

        print('starting unidirectional unweighted by dist sim')
        # and again, but only unidirectional activation deviations without wasserstein distance weight for distribution similarity
        # distances_by_feature_by_class_unsure_uni_unweighted, total_distances_unsure_uni_unweighted, distances_by_feature_by_class_weighted_unsure_uni_unweighted, total_distances_weighted_unsure_uni_unweighted, total_distances_unweighted_unsure_uni_unweighted, critical_feature_mismatches_unsure_uni_unweighted = distances_unsure_uni_unweighted
        #
        # DistributionDistancePlotter(distances_by_feature_by_class_unsure_uni_unweighted,
        #                             total_distances_unsure_uni_unweighted, no_weights_path_unsure_uni_unweighted,
        #                             total_distances_unweighted_unsure_uni_unweighted).plot()
        #
        # distances_by_feature_by_class_significant_uni_unweighted, total_distances_significant_uni_unweighted, distances_by_feature_by_class_weighted_significant_uni_unweighted, total_distances_weighted_significant_uni_unweighted, total_distances_unweighted_significant_uni_unweighted, critical_feature_mismatches_significant_uni_unweighted = distances_significant_uni_unweighted
        #
        # DistributionDistancePlotter(distances_by_feature_by_class_significant_uni_unweighted,
        #                             total_distances_significant_uni_unweighted,
        #                             no_weights_path_significant_uni_unweighted,
        #                             total_distances_unweighted_significant_uni_unweighted).plot()
        #
        # distances_by_feature_by_class_uni_unweighted_mad, total_distances_uni_unweighted_mad, distances_by_feature_by_class_weighted_uni_unweighted_mad, total_distances_weighted_uni_unweighted_mad, total_distances_unweighted_uni_unweighted_mad, critical_feature_mismatches_mad_uni_unweighted = distances_uni_unweighted_mad
        #
        # DistributionDistancePlotter(distances_by_feature_by_class_uni_unweighted_mad,
        #                             total_distances_uni_unweighted_mad, no_weights_path_uni_unweighted_mad,
        #                             total_distances_unweighted_uni_unweighted_mad).plot()

        distances_by_feature_by_class_uni_unweighted_qn, total_distances_uni_unweighted_qn, distances_by_feature_by_class_weighted_uni_unweighted_qn, total_distances_weighted_uni_unweighted_qn, total_distances_unweighted_uni_unweighted_qn, critical_feature_mismatches_qn_uni_unweighted = distances_uni_unweighted_qn

        DistributionDistancePlotter(distances_by_feature_by_class_uni_unweighted_qn,
                                    total_distances_uni_unweighted_qn, no_weights_path_uni_unweighted_qn,
                                    total_distances_unweighted_uni_unweighted_qn).plot()

        # self._plot_weight_vs_activation_diff(distances_by_feature_by_class_unsure_uni_unweighted,
        #                                      '_all_models_unidirectional_unweighted_by_dist_sim_logit')
        # self._plot_weight_vs_activation_diff(distances_by_feature_by_class_significant_uni_unweighted,
        #                                      '_significant_hurdle_models_unidirectional_unweighted_by_dist_sim_logit')
        # self._plot_weight_vs_activation_diff(distances_by_feature_by_class_uni_unweighted_mad,
        #                                      '_MAD_unidirectional_unweighted_by_dist_sim_logit')
        self._plot_weight_vs_activation_diff(distances_by_feature_by_class_uni_unweighted_qn,
                                             '_BASE_unidirectional_unweighted_by_dist_sim_logit')

    def _calculate_activation_over_features(self, act_dist_comparer, feature_triviality_by_feature_by_class_gen,
                                            feature_triviality_by_feature_by_class_test,
                                            feature_triviality_by_feature_by_class_train, wasserstein_dist_calculator,
                                            wasserstein_path):
        wasserstein_dist_calculator.calculate_mean_wasserstein_distance_by_class(feature_triviality_by_feature_by_class_test, feature_triviality_by_feature_by_class_gen)
        wasserstein_dist_calculator.calculate_mean_wasserstein_distance(feature_triviality_by_feature_by_class_test, feature_triviality_by_feature_by_class_gen)
        wasserstein_distances = wasserstein_dist_calculator.get_distances()

        plotter = WassersteinDistancePlotter(wasserstein_distances, wasserstein_path)
        plotter.plot()

        wasserstein_distances_within_datasets = wasserstein_dist_calculator.get_distances_between_classes_within_datasets()
        plotter.plot_distances_between_classes_within_datasets(wasserstein_distances_within_datasets)

        wasserstein_distances_between_datasets = wasserstein_dist_calculator.get_distance_between_classes_between_datasets()
        plotter.plot_distances_between_classes_between_datasets(wasserstein_distances_between_datasets)

        act_dist_comparer.print(
            feature_triviality_by_feature_by_class_train,
            feature_triviality_by_feature_by_class_test,
            feature_triviality_by_feature_by_class_gen,
        )

    def _calculate_activation_on_feature(self, act_dist_comparer, column, combined_dist_plotter,
                                         distribution_distance_calculator, feature,
                                         feature_triviality_by_feature_by_class_gen,
                                         feature_triviality_by_feature_by_class_test,
                                         feature_triviality_by_feature_by_class_train, gen_dist_plotter,
                                         test_dist_plotter, wasserstein_dist_calculator, logit=False):
        class_0_feature_values_test = self.df_test[self.df_test['Class'] == 0][column].tolist()
        class_1_feature_values_test = self.df_test[self.df_test['Class'] == 1][column].tolist()
        feature_values_test_combined = self.df_test[column].tolist()
        feature_values_test_tuple = class_0_feature_values_test, class_1_feature_values_test

        class_0_feature_values_gen = self.df_generated[self.df_generated['Class'] == 0][column].tolist()
        class_1_feature_values_gen = self.df_generated[self.df_generated['Class'] == 1][column].tolist()
        feature_values_gen_combined = self.df_generated[column].tolist()
        feature_values_gen_tuple = class_0_feature_values_gen, class_1_feature_values_gen

        class_0_feature_values_train = self.df_train[self.df_train['Class'] == 0][column].tolist()
        class_1_feature_values_train = self.df_train[self.df_train['Class'] == 1][column].tolist()
        feature_values_train_combined = self.df_train[column].tolist()
        feature_values_train_tuple = class_0_feature_values_train, class_1_feature_values_train

        self._calculate_on_data(act_dist_comparer, class_0_feature_values_gen, class_0_feature_values_test,
                                class_0_feature_values_train, class_1_feature_values_gen, class_1_feature_values_test,
                                class_1_feature_values_train, combined_dist_plotter, distribution_distance_calculator,
                                feature, feature_triviality_by_feature_by_class_gen,
                                feature_triviality_by_feature_by_class_test,
                                feature_triviality_by_feature_by_class_train, feature_values_gen_combined,
                                feature_values_gen_tuple, feature_values_test_combined, feature_values_test_tuple,
                                feature_values_train_combined, feature_values_train_tuple, gen_dist_plotter, logit,
                                test_dist_plotter, wasserstein_dist_calculator)

    def _calculate_logit_on_feature(self, act_dist_comparer, column, combined_dist_plotter,
                                         distribution_distance_calculator, feature,
                                         feature_triviality_by_feature_by_class_gen,
                                         feature_triviality_by_feature_by_class_test,
                                         feature_triviality_by_feature_by_class_train, gen_dist_plotter,
                                         test_dist_plotter, wasserstein_dist_calculator, logit=False):
        class_0_feature_values_test = self.df_logit_test[self.df_logit_test['Class'] == 0][column].tolist()
        class_1_feature_values_test = self.df_logit_test[self.df_logit_test['Class'] == 1][column].tolist()
        feature_values_test_combined = self.df_logit_test[column].tolist()
        feature_values_test_tuple = class_0_feature_values_test, class_1_feature_values_test

        class_0_feature_values_gen = self.df_logit_generated[self.df_logit_generated['Class'] == 0][column].tolist()
        class_1_feature_values_gen = self.df_logit_generated[self.df_logit_generated['Class'] == 1][column].tolist()
        feature_values_gen_combined = self.df_logit_generated[column].tolist()
        feature_values_gen_tuple = class_0_feature_values_gen, class_1_feature_values_gen

        class_0_feature_values_train = self.df_logit_train[self.df_logit_train['Class'] == 0][column].tolist()
        class_1_feature_values_train = self.df_logit_train[self.df_logit_train['Class'] == 1][column].tolist()
        feature_values_train_combined = self.df_logit_train[column].tolist()
        feature_values_train_tuple = class_0_feature_values_train, class_1_feature_values_train

        self._calculate_on_data(act_dist_comparer, class_0_feature_values_gen, class_0_feature_values_test,
                                class_0_feature_values_train, class_1_feature_values_gen, class_1_feature_values_test,
                                class_1_feature_values_train, combined_dist_plotter, distribution_distance_calculator,
                                feature, feature_triviality_by_feature_by_class_gen,
                                feature_triviality_by_feature_by_class_test,
                                feature_triviality_by_feature_by_class_train, feature_values_gen_combined,
                                feature_values_gen_tuple, feature_values_test_combined, feature_values_test_tuple,
                                feature_values_train_combined, feature_values_train_tuple, gen_dist_plotter, logit,
                                test_dist_plotter, wasserstein_dist_calculator)

    def _calculate_on_data(self, act_dist_comparer, class_0_feature_values_gen, class_0_feature_values_test,
                           class_0_feature_values_train, class_1_feature_values_gen, class_1_feature_values_test,
                           class_1_feature_values_train, combined_dist_plotter, distribution_distance_calculator,
                           feature, feature_triviality_by_feature_by_class_gen,
                           feature_triviality_by_feature_by_class_test, feature_triviality_by_feature_by_class_train,
                           feature_values_gen_combined, feature_values_gen_tuple, feature_values_test_combined,
                           feature_values_test_tuple, feature_values_train_combined, feature_values_train_tuple,
                           gen_dist_plotter, logit, test_dist_plotter, wasserstein_dist_calculator):
        # feature triviality
        is_feature_trivial_class_0_train = self._is_trivial_feature(class_0_feature_values_train, logit)
        is_feature_trivial_class_1_train = self._is_trivial_feature(class_1_feature_values_train, logit)

        feature_triviality_by_feature_by_class_train[feature][0] = is_feature_trivial_class_0_train
        feature_triviality_by_feature_by_class_train[feature][1] = is_feature_trivial_class_1_train

        is_feature_trivial_class_0_test = self._is_trivial_feature(class_0_feature_values_test, logit)
        is_feature_trivial_class_1_test = self._is_trivial_feature(class_1_feature_values_test, logit)

        feature_triviality_by_feature_by_class_test[feature][0] = is_feature_trivial_class_0_test
        feature_triviality_by_feature_by_class_test[feature][1] = is_feature_trivial_class_1_test

        is_feature_trivial_class_0_gen = self._is_trivial_feature(class_0_feature_values_gen, logit)
        is_feature_trivial_class_1_gen = self._is_trivial_feature(class_1_feature_values_gen, logit)

        feature_triviality_by_feature_by_class_gen[feature][0] = is_feature_trivial_class_0_gen
        feature_triviality_by_feature_by_class_gen[feature][1] = is_feature_trivial_class_1_gen

        # calculate and/or plot
        test_dist_plotter.plot(class_0_feature_values_test, class_1_feature_values_test, feature)
        gen_dist_plotter.plot(class_0_feature_values_gen, class_1_feature_values_gen, feature)

        combined_dist_plotter.plot(feature_values_test_tuple, feature_values_gen_tuple, feature)

        wasserstein_dist_calculator.calculate_by_feature(feature, feature_values_test_combined,
                                                         feature_values_gen_combined)
        wasserstein_dist_calculator.calculate_by_feature_by_class(feature, 0, class_0_feature_values_test,
                                                                  class_0_feature_values_gen)
        wasserstein_dist_calculator.calculate_by_feature_by_class(feature, 1, class_1_feature_values_test,
                                                                  class_1_feature_values_gen)

        wasserstein_dist_calculator.calculate_distances_between_classes_within_datasets(
            feature,
            class_0_feature_values_test,
            class_1_feature_values_test,
            class_0_feature_values_gen,
            class_1_feature_values_gen
        )
        wasserstein_dist_calculator.calculate_distance_between_classes_between_datasets(
            feature,
            class_0_feature_values_test,
            class_1_feature_values_test,
            class_0_feature_values_gen,
            class_1_feature_values_gen
        )
        wasserstein_dist_calculator.calculate_distances_between_classes_within_dataset(
            feature,
            class_0_feature_values_train,
            class_1_feature_values_train,
        )
        act_dist_comparer.compare(
            (feature_values_train_combined, feature_values_train_tuple),
            (feature_values_test_combined, feature_values_test_tuple),
            (feature_values_gen_combined, feature_values_gen_tuple),
        )
        distribution_distance_calculator.prepare(feature, class_0_feature_values_train, class_1_feature_values_train)
        distribution_distance_calculator.calculate(class_0_feature_values_test, class_0_feature_values_gen, feature, 0)
        distribution_distance_calculator.calculate(class_1_feature_values_test, class_1_feature_values_gen, feature, 1)

    def _make_activation_paths(self, logit=False):
        log = '_logit' if logit else ''

        wasserstein_path = f'{self.path_to_output_folder}{sep}wasserstein{log}'
        os.mkdir(wasserstein_path)

        ks_path = f'{self.path_to_output_folder}{sep}ks_test{log}'
        os.mkdir(ks_path)

        # no_weights_path_unsure = f'{self.path_to_output_folder}{sep}noweights_all_features{log}'
        # no_weights_path_significant = f'{self.path_to_output_folder}{sep}noweights_significant_models{log}'
        # weights_path_unsure = f'{self.path_to_output_folder}{sep}weighted_all_features{log}'
        # weights_path_significant = f'{self.path_to_output_folder}{sep}weighted_significant_models{log}'
        # os.mkdir(no_weights_path_unsure)
        # os.mkdir(no_weights_path_significant)
        # os.mkdir(weights_path_unsure)
        # os.mkdir(weights_path_significant)
        #
        # hurdle_model_stats_path = f'{self.path_to_output_folder}{sep}hurdle_model_stats{log}'
        # os.mkdir(hurdle_model_stats_path)
        #
        # critical_mismatch_path_unsure = f'{self.path_to_output_folder}{sep}critical_mismatches_unsure{log}'
        # critical_mismatch_path_significant = f'{self.path_to_output_folder}{sep}critical_mismatches_significant{log}'
        # os.mkdir(critical_mismatch_path_unsure)
        # os.mkdir(critical_mismatch_path_significant)

        # no_weights_path_mad = f'{self.path_to_output_folder}{sep}noweights_MAD{log}'
        # weights_path_mad = f'{self.path_to_output_folder}{sep}weighted_MAD{log}'
        no_weights_path_qn = f'{self.path_to_output_folder}{sep}noweights_BASE{log}'
        weights_path_qn = f'{self.path_to_output_folder}{sep}weighted_BASE{log}'
        # os.mkdir(no_weights_path_mad)
        # os.mkdir(weights_path_mad)
        os.mkdir(no_weights_path_qn)
        os.mkdir(weights_path_qn)

        # no_weights_path_unsure_uni_weighted = f'{self.path_to_output_folder}{sep}noweights_all_features_unidirectional{log}'
        # weights_path_unsure_uni_weighted = f'{self.path_to_output_folder}{sep}weighted_all_features_unidirectional{log}'
        # no_weights_path_significant_uni_weighted = f'{self.path_to_output_folder}{sep}noweights_significant_models_unidirectional{log}'
        # weights_path_significant_uni_weighted = f'{self.path_to_output_folder}{sep}weighted_significant_models_unidirectional{log}'
        # no_weights_path_unsure_uni_unweighted = f'{self.path_to_output_folder}{sep}noweights_all_features_unidirectional_unweighted_by_dist_sim{log}'
        # weights_path_unsure_uni_unweighted = f'{self.path_to_output_folder}{sep}weighted_all_features_unidirectional_unweighted_by_dist_sim{log}'
        # no_weights_path_significant_uni_unweighted = f'{self.path_to_output_folder}{sep}noweights_significant_models_unidirectional_unweighted_by_dist_sim{log}'
        # weights_path_significant_uni_unweighted = f'{self.path_to_output_folder}{sep}weighted_significant_models_unidirectional_unweighted_by_dist_sim{log}'
        # os.mkdir(no_weights_path_unsure_uni_weighted)
        # os.mkdir(weights_path_unsure_uni_weighted)
        # os.mkdir(no_weights_path_significant_uni_weighted)
        # os.mkdir(weights_path_significant_uni_weighted)
        # os.mkdir(no_weights_path_unsure_uni_unweighted)
        # os.mkdir(weights_path_unsure_uni_unweighted)
        # os.mkdir(no_weights_path_significant_uni_unweighted)
        # os.mkdir(weights_path_significant_uni_unweighted)

        # no_weights_path_uni_weighted_mad = f'{self.path_to_output_folder}{sep}noweights_unidirectional_MAD{log}'
        # weights_path_uni_weighted_mad = f'{self.path_to_output_folder}{sep}weighted_unidirectional_MAD{log}'
        no_weights_path_uni_weighted_qn = f'{self.path_to_output_folder}{sep}noweights_unidirectional_BASE{log}'
        weights_path_uni_weighted_qn = f'{self.path_to_output_folder}{sep}weighted_unidirectional_BASE{log}'
        # no_weights_path_uni_unweighted_mad = f'{self.path_to_output_folder}{sep}noweights_unidirectional_unweighted_by_dist_sim_MAD{log}'
        # weights_path_uni_unweighted_mad = f'{self.path_to_output_folder}{sep}weighted_unidirectional_unweighted_by_dist_sim_MAD{log}'
        no_weights_path_uni_unweighted_qn = f'{self.path_to_output_folder}{sep}noweights_unidirectional_unweighted_by_dist_sim_BASE{log}'
        weights_path_uni_unweighted_qn = f'{self.path_to_output_folder}{sep}weighted_unidirectional_unweighted_by_dist_sim_BASE{log}'
        # os.mkdir(no_weights_path_uni_weighted_mad)
        # os.mkdir(weights_path_uni_weighted_mad)
        os.mkdir(no_weights_path_uni_weighted_qn)
        os.mkdir(weights_path_uni_weighted_qn)
        # os.mkdir(no_weights_path_uni_unweighted_mad)
        # os.mkdir(weights_path_uni_unweighted_mad)
        os.mkdir(no_weights_path_uni_unweighted_qn)
        os.mkdir(weights_path_uni_unweighted_qn)

        return None, None, None, ks_path, None, no_weights_path_qn, None, None, None, None, no_weights_path_uni_unweighted_qn, None, no_weights_path_uni_weighted_qn, None, None, None, wasserstein_path, None, weights_path_qn, None, None, None, None, weights_path_uni_unweighted_qn, None, weights_path_uni_weighted_qn, None, None, None

    def _eval_on_logit_dataframes(self):
        self._calculate_logit_evaluation()

    def _calculate_logit_evaluation(self):
        feature_triviality_by_feature_by_class_train = defaultdict(dict)
        feature_triviality_by_feature_by_class_test = defaultdict(dict)
        feature_triviality_by_feature_by_class_gen = defaultdict(dict)

        # all output paths
        path_dist_plot_test, path_dist_plot_gen, path_comparison = self._mk_output_dirs(logit=True)

        critical_mismatch_path_significant, \
        critical_mismatch_path_unsure, \
        hurdle_model_stats_path, \
        ks_path, \
        no_weights_path_mad, \
        no_weights_path_qn, \
        no_weights_path_significant, \
        no_weights_path_significant_uni_unweighted, \
        no_weights_path_significant_uni_weighted, \
        no_weights_path_uni_unweighted_mad, \
        no_weights_path_uni_unweighted_qn, \
        no_weights_path_uni_weighted_mad, \
        no_weights_path_uni_weighted_qn, \
        no_weights_path_unsure, \
        no_weights_path_unsure_uni_unweighted, \
        no_weights_path_unsure_uni_weighted, \
        wasserstein_path, \
        weights_path_mad, \
        weights_path_qn, \
        weights_path_significant, \
        weights_path_significant_uni_unweighted, \
        weights_path_significant_uni_weighted, \
        weights_path_uni_unweighted_mad, \
        weights_path_uni_unweighted_qn, \
        weights_path_uni_weighted_mad, \
        weights_path_uni_weighted_qn, \
        weights_path_unsure, \
        weights_path_unsure_uni_unweighted, \
        weights_path_unsure_uni_weighted = self._make_activation_paths(logit=True)

        print('Output paths for logits done')

        # calculators and plotters
        test_dist_plotter = LogitDistributionPlotter(path_dist_plot_test)
        gen_dist_plotter = LogitDistributionPlotter(path_dist_plot_gen)
        combined_dist_plotter = LogitCombinedDistributionPlotter(
            path_comparison,
        )

        wasserstein_dist_calculator = LogitWassersteinDistanceCalculator()
        self.logit_wasserstein_distances_within_train_set = wasserstein_dist_calculator.distances_within_train_set

        act_dist_comparer = LogitActivationDistributionComparer(ks_path)

        distribution_distance_calculator = LogitDistributionDistanceCalculator(
                                                                          self._best_logit_distribution_models,
                                                                          self.df_logit_feature_count,
                                                                          self.logit_wasserstein_distances_within_train_set,
        )

        print('Starting to iterate over logit data')
        # data
        for feature in range(self.df_logit_feature_count):
            column = f'Logit_{feature}'

            self._calculate_logit_on_feature(
                act_dist_comparer,
                column,
                combined_dist_plotter,
                distribution_distance_calculator,
                feature,
                feature_triviality_by_feature_by_class_gen,
                feature_triviality_by_feature_by_class_test,
                feature_triviality_by_feature_by_class_train,
                gen_dist_plotter,
                test_dist_plotter,
                wasserstein_dist_calculator,
            )
            print('logit calculation on feature done')

        self._calculate_activation_over_features(
            act_dist_comparer,
            feature_triviality_by_feature_by_class_gen,
            feature_triviality_by_feature_by_class_test,
            feature_triviality_by_feature_by_class_train,
            wasserstein_dist_calculator,
            wasserstein_path,
        )
        print('Logit calculation over all features done')

        self._calculate_pair_logit_distances(
            critical_mismatch_path_significant,
            critical_mismatch_path_unsure,
            distribution_distance_calculator,
            hurdle_model_stats_path,
            no_weights_path_mad,
            no_weights_path_qn,
            no_weights_path_significant,
            no_weights_path_significant_uni_unweighted,
            no_weights_path_significant_uni_weighted,
            no_weights_path_uni_unweighted_mad,
            no_weights_path_uni_unweighted_qn,
            no_weights_path_uni_weighted_mad,
            no_weights_path_uni_weighted_qn,
            no_weights_path_unsure,
            no_weights_path_unsure_uni_unweighted,
            no_weights_path_unsure_uni_weighted,
        )
        print('Logit pair distance calculations done')

    def _get_feature_count(self, df):
        feature_columns = [col for col in df if col.startswith('feature')]
        return len(feature_columns)

    def _is_trivial_feature(self, class_values, logit):
        if logit:
            return False
        else:
            return not any(filter(lambda v: v > 0, class_values))

    def _plot_weight_vs_activation_diff(self, distances_by_feature_by_class, path_extension=''):
        plot_path = f'{self.path_to_output_folder}{sep}weight_activation_plots{path_extension}'
        os.mkdir(plot_path)

        WeightActivationPlotter(self.abs_weight_by_feature_by_class, self.abs_weight_by_feature,
                                distances_by_feature_by_class, plot_path, weight_by_feature_by_class=self.weight_by_feature_by_class).plot()

        plot_path = f'{self.path_to_output_folder}{sep}weight_activation_plots{path_extension}_zero_weights_included'
        os.mkdir(plot_path)

        WeightActivationPlotter(self.abs_weight_by_feature_by_class, self.abs_weight_by_feature,
                                distances_by_feature_by_class,
                                plot_path, plot_zero_dist=True, weight_by_feature_by_class=self.weight_by_feature_by_class).plot()

    def _load_best_distribution_models(self):
        df = pd.read_csv(self.path_to_distribution_models, sep=';')
        return df

    def _load_best_logit_distribution_models(self):
        df = pd.read_csv(self.path_to_logit_models, sep=';')
        return df

    def _plot_db_distances(self):
        svm = self._load_svm()

        svm_dist_calc_test = SVMDBDistanceCalculator(svm, [self.df_test], ['Test'])
        svm_dist_calc_gen = SVMDBDistanceCalculator(svm, [self.df_generated], ['Generated'])

        train_name_dist_avg_std = SVMDBDistanceCalculator(svm, [self.df_train], ['Train']).calculate()
        test_name_dist_avg_std = svm_dist_calc_test.calculate()
        generated_name_dist_avg_std = svm_dist_calc_gen.calculate()

        svm_folder = f'{self.path_to_output_folder}//svm'
        os.mkdir(svm_folder)

        # Step 1: simply plot test, generated distances
        test_path = f'{svm_folder}//svm_test'
        gen_path = f'{svm_folder}//svm_generated'

        os.mkdir(test_path)
        os.mkdir(gen_path)

        SVMDBDistancePlotter(test_name_dist_avg_std, test_path).plot()
        SVMDBDistancePlotter(generated_name_dist_avg_std, gen_path).plot()

        # Step 2: check if distance distributions are the same
        SVMDBDistanceDistributionComparer(train_name_dist_avg_std, test_name_dist_avg_std, svm_folder, 'train_test').compare()
        SVMDBDistanceDistributionComparer(train_name_dist_avg_std, generated_name_dist_avg_std, svm_folder, 'train_generated').compare()
        SVMDBDistanceDistributionComparer(test_name_dist_avg_std, generated_name_dist_avg_std, svm_folder, 'test_generated').compare()

        # Step 3: calculate changes in distances between test set and generated set
        distance_differences = SVMDBDistanceComparer(train_name_dist_avg_std, test_name_dist_avg_std,
                                                     generated_name_dist_avg_std, svm_folder).compare()
        SVMDBDistanceComparisonPlotter(distance_differences, svm_folder).plot()

        # Step 4: calculate unidirectional changes in distances between test set and generated set
        svm_folder_undirectional = f'{svm_folder}_unidirectional'
        os.mkdir(svm_folder_undirectional)

        unidir_distance_comparer = UnidirectionalSVMDBDistanceComparer(train_name_dist_avg_std, test_name_dist_avg_std,
                                                       generated_name_dist_avg_std, svm_folder)
        distance_differences = unidir_distance_comparer.compare()
        SVMDBDistanceComparisonPlotter(distance_differences, svm_folder_undirectional).plot()


        # check that both original and generated are on same side of SVM DB
        test_name_dist_avg_std_signed = svm_dist_calc_test.calculate_signed()
        generated_name_dist_avg_std_signed = svm_dist_calc_gen.calculate_signed()

        unidir_distance_comparer = UnidirectionalSVMDBDistanceComparer(train_name_dist_avg_std, test_name_dist_avg_std_signed,
                                                                       generated_name_dist_avg_std_signed, svm_folder)
        count_of_same_sign_before_and_after = unidir_distance_comparer.compare_distance_sign()

        lines = [
            f'For {name}: From {d[1]} total, {d[0]} did not change sign with regard to SVM DB.\n'
            for name, d in count_of_same_sign_before_and_after.items()
        ]
        with open(f'{svm_folder}{sep}sign_changes.txt', 'w') as file:
            file.writelines(lines)

    def _load_svm(self):
        svm = load(self.svm_path)
        return svm

    def _mk_output_dirs(self, logit=False):
        test_dir = f'{self.path_to_output_folder}{sep}test{"_logit" if logit else ""}'
        generated_dir = f'{self.path_to_output_folder}{sep}generated{"_logit" if logit else ""}'
        comparison_dir = f'{self.path_to_output_folder}{sep}comparison{"_logit" if logit else ""}'

        os.mkdir(test_dir)
        os.mkdir(generated_dir)
        os.mkdir(comparison_dir)

        return test_dir, generated_dir, comparison_dir

    def _load_train_distributions(self):
        df = pd.read_csv(self.path_to_distribution_csv, sep=';')
        df2 = pd.read_csv(self.path_to_logit_csv, sep=';')
        return df, df2