from os.path import sep

from scipy import stats
import numpy as np


class ActivationDistributionComparer:
    
    def __init__(self, path):
        self.output_folder = path

        self.train_test_p_values_both = []
        self.train_test_p_values_class_0 = []
        self.train_test_p_values_class_1 = []
        self.train_generated_p_values_both = []
        self.train_generated_p_values_class_0 = []
        self.train_generated_p_values_class_1 = []
        self.test_generated_p_values_both = []
        self.test_generated_p_values_class_0 = []
        self.test_generated_p_values_class_1 = []

        np.random.seed(42)

        self.name = 'feature'

    def print(
            self,
            feature_triviality_by_feature_by_class_train,
            feature_triviality_by_feature_by_class_test,
            feature_triviality_by_feature_by_class_gen,
    ):
        alpha = 0.05

        lines = [
            f'A p value <= {alpha} indicates that the distributions are not the same.\n',
            '\n',
        ]

        train_test_trivial_feature_both = 0
        train_test_trivial_feature_class_0 = 0
        train_test_trivial_feature_class_1 = 0
        train_test_both_same = 0
        train_test_class_0_same = 0
        train_test_class_1_same = 0

        train_generated_trivial_feature_both = 0
        train_generated_trivial_feature_class_0 = 0
        train_generated_trivial_feature_class_1 = 0
        train_generated_both_same = 0
        train_generated_class_0_same = 0
        train_generated_class_1_same = 0

        test_generated_trivial_feature_both = 0
        test_generated_trivial_feature_class_0 = 0
        test_generated_trivial_feature_class_1 = 0
        test_generated_both_same = 0
        test_generated_class_0_same = 0
        test_generated_class_1_same = 0

        for feature in range(len(self.train_test_p_values_both)):
            train_test_p_value_both = self.train_test_p_values_both[feature]
            train_test_p_value_class_0 = self.train_test_p_values_class_0[feature]
            train_test_p_value_class_1 = self.train_test_p_values_class_1[feature]

            train_generated_p_value_both = self.train_generated_p_values_both[feature]
            train_generated_p_value_class_0 = self.train_generated_p_values_class_0[feature]
            train_generated_p_value_class_1 = self.train_generated_p_values_class_1[feature]

            test_generated_p_value_both = self.test_generated_p_values_both[feature]
            test_generated_p_value_class_0 = self.test_generated_p_values_class_0[feature]
            test_generated_p_value_class_1 = self.test_generated_p_values_class_1[feature]

            lines.extend([
                f'{self.name} {feature}:\n',
                f'------------------------\n',
                f'P-Value train vs. test (both classes): {train_test_p_value_both} -- {self._same_label(alpha, train_test_p_value_both)}\n',
                f'P-Value train vs. test (class 0): {train_test_p_value_class_0} -- {self._same_label(alpha, train_test_p_value_class_0)}\n',
                f'P-Value train vs. test (class 1): {train_test_p_value_class_1} -- {self._same_label(alpha, train_test_p_value_class_1)}\n',
                '\n',
                f'P-Value train vs. generated (both classes): {train_generated_p_value_both} -- {self._same_label(alpha, train_generated_p_value_both)}\n',
                f'P-Value train vs. generated (class 0): {train_generated_p_value_class_0} -- {self._same_label(alpha, train_generated_p_value_class_0)}\n',
                f'P-Value train vs. generated (class 1): {train_generated_p_value_class_1} -- {self._same_label(alpha, train_generated_p_value_class_1)}\n',
                '\n',
                f'P-Value test vs. generated (both classes): {test_generated_p_value_both} -- {self._same_label(alpha, test_generated_p_value_both)}\n',
                f'P-Value test vs. generated (class 0): {test_generated_p_value_class_0} -- {self._same_label(alpha, test_generated_p_value_class_0)}\n',
                f'P-Value test vs. generated (class 1): {test_generated_p_value_class_1} -- {self._same_label(alpha, test_generated_p_value_class_1)}\n',
                f'------------------------\n',
                '\n',
            ])

            train_test_p_value_both_same = self._same(alpha, train_test_p_value_both)
            train_test_p_value_class_0_same = self._same(alpha, train_test_p_value_class_0)
            train_test_p_value_class_1_same = self._same(alpha, train_test_p_value_class_1)

            train_test_0_trivial = self._is_trivial(feature_triviality_by_feature_by_class_train, feature_triviality_by_feature_by_class_test, feature, 0)
            train_test_1_trivial = self._is_trivial(feature_triviality_by_feature_by_class_train, feature_triviality_by_feature_by_class_test, feature, 1)
            train_test_both_trivial = train_test_0_trivial and train_test_1_trivial
            
            train_test_trivial_feature_class_0 += train_test_0_trivial
            train_test_trivial_feature_class_1 += train_test_1_trivial
            train_test_trivial_feature_both += train_test_both_trivial
            train_test_both_same += train_test_p_value_both_same
            train_test_class_0_same += train_test_p_value_class_0_same
            train_test_class_1_same += train_test_p_value_class_1_same

            train_generated_p_value_both_same = self._same(alpha, train_generated_p_value_both)
            train_generated_p_value_class_0_same = self._same(alpha, train_generated_p_value_class_0)
            train_generated_p_value_class_1_same = self._same(alpha, train_generated_p_value_class_1)

            train_generated_0_trivial = self._is_trivial(feature_triviality_by_feature_by_class_train, feature_triviality_by_feature_by_class_gen, feature, 0)
            train_generated_1_trivial = self._is_trivial(feature_triviality_by_feature_by_class_train, feature_triviality_by_feature_by_class_gen, feature, 1)
            train_generated_both_trivial = train_generated_0_trivial and train_generated_1_trivial
            
            train_generated_trivial_feature_class_0 += train_generated_0_trivial
            train_generated_trivial_feature_class_1 += train_generated_1_trivial
            train_generated_trivial_feature_both += train_generated_both_trivial
            train_generated_both_same += train_generated_p_value_both_same
            train_generated_class_0_same += train_generated_p_value_class_0_same
            train_generated_class_1_same += train_generated_p_value_class_1_same

            test_generated_p_value_both_same = self._same(alpha, test_generated_p_value_both)
            test_generated_p_value_class_0_same = self._same(alpha, test_generated_p_value_class_0)
            test_generated_p_value_class_1_same = self._same(alpha, test_generated_p_value_class_1)

            test_generated_0_trivial = self._is_trivial(feature_triviality_by_feature_by_class_test, feature_triviality_by_feature_by_class_gen, feature, 0)
            test_generated_1_trivial = self._is_trivial(feature_triviality_by_feature_by_class_test, feature_triviality_by_feature_by_class_gen, feature, 1)
            test_generated_both_trivial = test_generated_0_trivial and test_generated_1_trivial
            
            test_generated_trivial_feature_class_0 += test_generated_0_trivial 
            test_generated_trivial_feature_class_1 += test_generated_1_trivial
            test_generated_trivial_feature_both += test_generated_both_trivial
            test_generated_both_same += test_generated_p_value_both_same
            test_generated_class_0_same += test_generated_p_value_class_0_same
            test_generated_class_1_same += test_generated_p_value_class_1_same

        total_features = len(self.train_test_p_values_both)
        
        lines.extend([
            '\n',
            '\n',
            f'Overview of how many {self.name}s are the same (non-trivially):\n',
            f'------------------------\n',
            '\n',
            'Train vs. Test:\n',
            f'trivial {self.name}s (trivial in both classes): {train_test_trivial_feature_both}/{total_features} -- {((train_test_trivial_feature_both/total_features) * 100):.2f}%\n',
            f'Same distribution (in both classes) for {self.name}s: {train_test_both_same}/{total_features} -- {((train_test_both_same/total_features) * 100):.2f}%\n',
            f'Same distribution (in both classes) for {self.name}s -- non-trivial: {train_test_both_same - train_test_trivial_feature_both}/{total_features - train_test_trivial_feature_both} -- {(((train_test_both_same-train_test_trivial_feature_both) / (total_features-train_test_trivial_feature_both)) * 100):.2f}%\n',
            '\n',
            f'trivial {self.name}s (class 0): {train_test_trivial_feature_class_0}/{total_features} -- {((train_test_trivial_feature_class_0/total_features) * 100):.2f}%\n',
            f'Same distribution (class 0) for {self.name}s: {train_test_class_0_same}/{total_features} -- {((train_test_class_0_same/total_features) * 100):.2f}%\n',
            f'Same distribution (class 0) for {self.name}s -- non-trivial: {train_test_class_0_same-train_test_trivial_feature_class_0}/{total_features-train_test_trivial_feature_class_0} -- {(((train_test_class_0_same-train_test_trivial_feature_class_0) / (total_features-train_test_trivial_feature_class_0)) * 100):.2f}%\n',
            '\n',
            f'trivial {self.name}s (class 1): {train_test_trivial_feature_class_1}/{total_features} -- {((train_test_trivial_feature_class_1/total_features) * 100):.2f}%\n',
            f'Same distribution (class 1) for {self.name}s: {train_test_class_1_same}/{total_features} -- {((train_test_class_1_same/total_features) * 100):.2f}%\n',
            f'Same distribution (class 1) for {self.name}s -- non-trivial: {train_test_class_1_same-train_test_trivial_feature_class_1}/{total_features-train_test_trivial_feature_class_1} -- {(((train_test_class_1_same-train_test_trivial_feature_class_1) / (total_features-train_test_trivial_feature_class_1)) * 100):.2f}%\n',
            '\n',
            f'------------------------\n',
            '\n',
            'Train vs. Generated:\n',
            f'trivial {self.name}s (trivial in both classes): {train_generated_trivial_feature_both}/{total_features} -- {((train_generated_trivial_feature_both/total_features) * 100):.2f}%\n',
            f'Same distribution (in both classes) for {self.name}s: {train_generated_both_same}/{total_features} -- {((train_generated_both_same/total_features) * 100):.2f}%\n',
            f'Same distribution (in both classes) for {self.name}s -- non-trivial: {train_generated_both_same-train_generated_trivial_feature_both}/{total_features-train_generated_trivial_feature_both} -- {(((train_generated_both_same-train_generated_trivial_feature_both) / (total_features-train_generated_trivial_feature_both)) * 100):.2f}%\n',
            '\n',
            f'trivial {self.name}s (class 0): {train_generated_trivial_feature_class_0}/{total_features} -- {((train_generated_trivial_feature_class_0/total_features) * 100):.2f}%\n',
            f'Same distribution (class 0) for {self.name}s: {train_generated_class_0_same}/{total_features} -- {((train_generated_class_0_same/total_features) * 100):.2f}%\n',
            f'Same distribution (class 0) for {self.name} -- non-trivial: {train_generated_class_0_same-train_generated_trivial_feature_class_0}/{total_features-train_generated_trivial_feature_class_0} -- {(((train_generated_class_0_same-train_generated_trivial_feature_class_0) / (total_features-train_generated_trivial_feature_class_0)) * 100):.2f}%\n',
            '\n',
            f'trivial {self.name}s (class 1): {train_generated_trivial_feature_class_1}/{total_features} -- {((train_generated_trivial_feature_class_1/total_features) * 100):.2f}%\n',
            f'Same distribution (class 1) for {self.name}s: {train_generated_class_1_same}/{total_features} -- {((train_generated_class_1_same/total_features) * 100):.2f}%\n',
            f'Same distribution (class 1) for {self.name}s -- non-trivial: {train_generated_class_1_same-train_generated_trivial_feature_class_1}/{total_features-train_generated_trivial_feature_class_1} -- {(((train_generated_class_1_same-train_generated_trivial_feature_class_1) / (total_features-train_generated_trivial_feature_class_1)) * 100):.2f}%\n',
            '\n',
            f'------------------------\n',
            '\n',
            'Test vs. Generated:\n'
            f'trivial {self.name}s (trivial in both classes): {test_generated_trivial_feature_both}/{total_features} -- {((test_generated_trivial_feature_both/total_features) * 100):.2f}%\n',
            f'Same distribution (in both classes) for {self.name}s: {test_generated_both_same}/{total_features} -- {((test_generated_both_same/total_features) * 100):.2f}%\n',
            f'Same distribution (in both classes) for {self.name}s -- non-trivial: {test_generated_both_same-test_generated_trivial_feature_both}/{total_features-test_generated_trivial_feature_both} -- {(((test_generated_both_same-test_generated_trivial_feature_both) / (total_features-test_generated_trivial_feature_both)) * 100):.2f}%\n',
            '\n',
            f'trivial {self.name}s (class 0): {test_generated_trivial_feature_class_0}/{total_features} -- {((test_generated_trivial_feature_class_0/total_features) * 100):.2f}%\n',
            f'Same distribution (class 0) for {self.name}s: {test_generated_class_0_same}/{total_features} -- {((test_generated_class_0_same/total_features) * 100):.2f}%\n',
            f'Same distribution (class 0) for {self.name}s -- non-trivial: {test_generated_class_0_same-test_generated_trivial_feature_class_0}/{total_features-test_generated_trivial_feature_class_0} -- {(((test_generated_class_0_same-test_generated_trivial_feature_class_0) / (total_features-test_generated_trivial_feature_class_0)) * 100):.2f}%\n',
            '\n',
            f'trivial {self.name}s (class 1): {test_generated_trivial_feature_class_1}/{total_features} -- {((test_generated_trivial_feature_class_1/total_features) * 100):.2f}%\n',
            f'Same distribution (class 1) for {self.name}s: {test_generated_class_1_same}/{total_features} -- {((test_generated_class_1_same/total_features) * 100):.2f}%\n',
            f'Same distribution (class 1) for {self.name}s -- non-trivial: {test_generated_class_1_same-test_generated_trivial_feature_class_1}/{total_features-test_generated_trivial_feature_class_1} -- {(((test_generated_class_1_same-test_generated_trivial_feature_class_1) / (total_features-test_generated_trivial_feature_class_1)) * 100):.2f}%\n',
        ])

        with open(f'{self.output_folder}{sep}Two_sided_kolmogorov_smirnov_on_activation_distributions.txt', 'w') as file:
            file.writelines(lines)

    def _same_label(self, alpha, value):
        return "same" if self._same(alpha, value) else "not same"

    def _same(self, alpha, value):
        return value > alpha

    def compare(
            self,
            train_data,
            test_data,
            gen_data,
    ):
        train_both, train_tuple = train_data
        train_0, train_1 = train_tuple
        test_both, test_tuple = test_data
        test_0, test_1 = test_tuple
        gen_both, gen_tuple = gen_data
        gen_0, gen_1 = gen_tuple

        combinations = [
            # train test
            (
                train_both,
                test_both,
                self.train_test_p_values_both,
                train_0,
                test_0,
                self.train_test_p_values_class_0,
                train_1,
                test_1,
                self.train_test_p_values_class_1,
                False,
            ),
            # train gen
            (
                train_both,
                gen_both,
                self.train_generated_p_values_both,
                train_0,
                gen_0,
                self.train_generated_p_values_class_0,
                train_1,
                gen_1,
                self.train_generated_p_values_class_1,
                False,
            ),
            # test gen
            (
                test_both,
                gen_both,
                self.test_generated_p_values_both,
                test_0,
                gen_0,
                self.test_generated_p_values_class_0,
                test_1,
                gen_1,
                self.test_generated_p_values_class_1,
                True
            ),
        ]

        for (d1_both, d2_both, p_both, d1_0, d2_0, p_0, d1_1, d2_1, p_1, dependent) in combinations:

            if dependent:
                test = stats.wilcoxon
            else:
                test = stats.ks_2samp

            try:
                _, p_value_both_classes = test(d1_both, d2_both)
            except:
                p_value_both_classes = 100 # wilcoxon does not work for zero features

            p_both.append(p_value_both_classes)

            try:
                _, p_value_class_0 = test(d1_0, d2_0)
            except:
                p_value_class_0 = 100 # wilcoxon does not work for zero features
            p_0.append(p_value_class_0)

            try:
                _, p_value_class_1 = test(d1_1, d2_1)
            except:
                p_value_class_1 = 100 # wilcoxon does not work for zero features
            p_1.append(p_value_class_1)

    def _is_trivial(self, feature_triviality_by_feature_by_class_a, feature_triviality_by_feature_by_class_b,
                    feature, label):
        return feature_triviality_by_feature_by_class_a[feature][label] and feature_triviality_by_feature_by_class_b[feature][label]

