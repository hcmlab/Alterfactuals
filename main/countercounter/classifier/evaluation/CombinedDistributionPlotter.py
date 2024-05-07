from main.countercounter.classifier.evaluation.DistributionPlotter import DistributionPlotter


class CombinedDistributionPlotter(DistributionPlotter):

    def __init__(self, path_to_output_folder):
        super().__init__(path_to_output_folder)
        self.name = 'Feature'

    def _plot_class_distributions_by_features(self, feature_values_test, feature_values_gen, feature):
        labels = ['Test', 'Generated']
        data_class_0 = []
        data_class_1 = []

        class_0_f_v_test, class_1_f_v_test = feature_values_test
        class_0_f_v_gen, class_1_f_v_gen = feature_values_gen

        data_class_0.append(class_0_f_v_test)
        data_class_0.append(class_0_f_v_gen)

        data_class_1.append(class_1_f_v_test)
        data_class_1.append(class_1_f_v_gen)

        self._print_distributions(data_class_0, data_class_1, labels, feature)

    def _print_distributions(self, class_0_feature_values, class_1_feature_values, labels, feature_idx):
        self._print_joint_distribution(
            class_0_feature_values,
            labels,
            f'{self.name} Comparison feature {feature_idx} Class 0',
        )

        self._print_joint_distribution(
            class_1_feature_values,
            labels,
            f'{self.name} Comparison feature {feature_idx} Class 1',
        )

        joint_labels = []
        joint_data = []

        for idx, label in enumerate(labels):
            class_0 = class_0_feature_values[idx]
            class_1 = class_1_feature_values[idx]

            joint_labels.append(f'{label} Class 0')
            joint_data.append(class_0)

            joint_labels.append(f'{label} Class 1')
            joint_data.append(class_1)

        self._print_joint_distribution(
            joint_data,
            joint_labels,
            f'{self.name} {feature_idx} Activations Comparison'
        )

    def plot(self, feature_values_test, feature_values_gen, feature):
        self._plot_class_distributions_by_features(feature_values_test, feature_values_gen, feature)