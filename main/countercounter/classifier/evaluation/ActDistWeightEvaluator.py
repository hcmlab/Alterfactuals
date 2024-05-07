from os.path import sep

from main.countercounter.classifier.emotion_classifier.CustomNets import SoftmaxLogitWrapper, CustomNetSmallGAPLogits
import pandas as pd
from scipy.stats import wasserstein_distance
from scipy.stats import spearmanr

from main.countercounter.gan._execution.evaluation.activationdistribution.WeightActivationCalculator import \
    WeightActivationCalculator


class ActDistWeightEvaluator:

    def __init__(self, setup, path_to_output_folder, activation_csv):
        if not isinstance(setup.model, SoftmaxLogitWrapper):
            raise NotImplementedError

        self.model: CustomNetSmallGAPLogits = setup.model.model
        self.model.eval()

        self.path_to_output_folder = path_to_output_folder

        self.df_train = pd.read_csv(activation_csv, sep=';')

    def evaluate(self):
        feature_count = self._get_feature_count()

        dist_by_feature, feature_triviality = self._get_wasserstein_distance_by_feature(feature_count)
        abs_weight_by_feature_by_class, abs_weight_by_feature = WeightActivationCalculator(None, None, feature_count=feature_count, cnn=self.model).calculate()
        self._save_correlation(feature_count, dist_by_feature, abs_weight_by_feature, abs_weight_by_feature_by_class, feature_triviality)

    def _get_wasserstein_distance_by_feature(self, feature_count):
        class_0_data = self.df_train[self.df_train['Class'] == 0]
        class_1_data = self.df_train[self.df_train['Class'] == 1]

        dist_by_feature = {}
        feature_triviality = {}

        for feature in range(feature_count):
            column = f'feature_{feature}'

            feature_0 = class_0_data[column].tolist()
            feature_1 = class_1_data[column].tolist()

            dist = wasserstein_distance(feature_0, feature_1)
            dist_by_feature[feature] = dist

            if any(filter(lambda v: v > 0, feature_0)) or any(filter(lambda v: v > 0, feature_1)):
                feature_triviality[feature] = False
            else:
                feature_triviality[feature] = True

        return dist_by_feature, feature_triviality

    def _save_correlation(self, feature_count, dist_by_feature, abs_weight_by_feature, abs_weight_by_feature_by_class, feature_triviality):

        lines = [
            'Similarity of act. distributions (via wasserstein distance) for classes 0 and 1 vs. abs weight per feature analysed with the Spearman correlation\n',
            '\n',
            'Coefficient 0: no correlation, -1, +1 exact monotonic relationship\n',
            'A p-Value <= 0.05 means that zero hypothesis (no correlation) is false.\n',
            '(Technically, switching the hypotheses would be better ...)\n',
            'ATT: p-Value does not reveal how strong the relation is (--> coefficient).\n'
            '\n',
        ]

        weights_0 = []
        weights_1 = []
        weights_both = []

        distances = []

        weights_0_no_trivial = []
        weights_1_no_trivial = []
        weights_both_no_trivial = []

        distances_no_trivial = []

        for feature in range(feature_count):
            dist = dist_by_feature[feature]

            weight_0 = abs_weight_by_feature_by_class[feature][0]
            weight_1 = abs_weight_by_feature_by_class[feature][1]
            weight_both = abs_weight_by_feature[feature]

            distances.append(dist)
            weights_0.append(weight_0)
            weights_1.append(weight_1)
            weights_both.append(weight_both)

            if feature_triviality[feature]:
                continue

            weights_0_no_trivial.append(weight_0)
            weights_1_no_trivial.append(weight_1)
            weights_both_no_trivial.append(weight_both)
            distances_no_trivial.append(dist)

        coeff_0, p_value_0 = spearmanr(weights_0, distances)
        coeff_1, p_value_1 = spearmanr(weights_1, distances)
        coeff_both, p_value_both = spearmanr(weights_both, distances)

        lines.extend([
            f'---------------------\n',
            f'All features\n',
            f'Weight for class 0 vs. dist -- Coefficient: {coeff_0} -- P-Value: {p_value_0}\n',
            f'Weight for class 1 vs. dist -- Coefficient: {coeff_1} -- P-Value: {p_value_1}\n',
            f'Weight for both classes (mean) vs. dist -- Coefficient: {coeff_both} -- P-Value: {p_value_both}\n',
            f'---------------------\n',
            '\n',
        ])

        coeff_0, p_value_0 = spearmanr(weights_0_no_trivial, distances_no_trivial)
        coeff_1, p_value_1 = spearmanr(weights_1_no_trivial, distances_no_trivial)
        coeff_both, p_value_both = spearmanr(weights_both_no_trivial, distances_no_trivial)

        lines.extend([
            f'---------------------\n',
            f'Non-trivial features:\n',
            f'Weight for class 0 vs. dist -- Coefficient: {coeff_0} -- P-Value: {p_value_0}\n',
            f'Weight for class 1 vs. dist -- Coefficient: {coeff_1} -- P-Value: {p_value_1}\n',
            f'Weight for both classes (mean) vs. dist -- Coefficient: {coeff_both} -- P-Value: {p_value_both}\n',
            f'---------------------\n',
            '\n',
        ])

        with open(f'{self.path_to_output_folder}{sep}spearman_correlation_act_dist_sim_weights.txt', 'w') as file:
            file.writelines(lines)

    def _get_feature_count(self):
        feature_columns = [col for col in self.df_train if col.startswith('feature')]
        return len(feature_columns)