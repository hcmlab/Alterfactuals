from os.path import sep
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")
import numpy as np


class WassersteinDistancePlotter:

    def __init__(self, wasserstein_distances, path):
        self.wasserstein_distance_by_feature_by_class, \
        self.wasserstein_distance_by_feature, \
        self.mean_wasserstein_distance_by_class, \
        self.mean_wasserstein_distance, \
        self.std_wasserstein_distance = wasserstein_distances

        self.output_folder = path

    def plot(self):
        self._print_distances()
        self._plot_mean_distance_by_class()
        self.plot_distance_by_feature()
        self.plot_distance_by_feature_by_class()

    def _print_distances(self):
        lines = [
            f'Avg Wasserstein distance over all classes and features: {self.mean_wasserstein_distance:.4f}\n',
            '\n'
        ] + [
            f'Std Wasserstein distance over all classes and features: {self.std_wasserstein_distance:.4f}\n',
            '\n'
        ] + [
            f'Avg Wasserstein distance for class {label}: {self.mean_wasserstein_distance_by_class[label]:.4f}\n' for label in sorted(self.mean_wasserstein_distance_by_class)
        ] + [
            '\n'
        ] + [
            f'Wasserstein distance for feature {feature}: {self.wasserstein_distance_by_feature[feature]:.4f}\n' for feature in sorted(self.wasserstein_distance_by_feature)
        ] + [
            '\n'
        ] + [
            f'Wasserstein distance for feature {feature} and class {label}: {self.wasserstein_distance_by_feature_by_class[feature][label]:.4f}\n'
            for feature in sorted(self.wasserstein_distance_by_feature) for label in sorted(self.wasserstein_distance_by_feature_by_class[feature])
        ]

        with open(f'{self.output_folder}{sep}wasserstein_mean_distances.txt', 'w') as file:
            file.writelines(lines)

    def _plot_mean_distance_by_class(self):
        fig, ax = plt.subplots()

        x = np.arange(len(self.mean_wasserstein_distance_by_class.keys()))
        y = list(map(lambda k: self.mean_wasserstein_distance_by_class[k], sorted(self.mean_wasserstein_distance_by_class.keys())))

        bar = ax.bar(x, y)
        ax.set_title('Avg. Wasserstein Distance Original - Generated Distributions by Class')
        ax.bar_label(bar, padding=3, fmt='%.2f')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{idx}' for idx in x])

        ax.set_xlabel('Class')
        ax.set_ylabel('Avg. Wasserstein Distance')

        plt.savefig(f'{self.output_folder}{sep}Wasserstein_distance_by_class')
        plt.close()

    def plot_distance_by_feature(self):
        fig, ax = plt.subplots()

        fig.set_figwidth(30)
        fig.set_figheight(15)

        x = np.arange(len(self.wasserstein_distance_by_feature.keys()))
        y = list(map(lambda k: self.wasserstein_distance_by_feature[k], sorted(self.wasserstein_distance_by_feature.keys())))

        bar = ax.bar(x, y)
        ax.set_title('Wasserstein Distance Original - Generated Distributions by Feature')
        ax.bar_label(bar, padding=3, fmt='%.2f')
        ax.set_xticks(x)
        ax.set_xticklabels([f'f_{idx}' for idx in x])

        ax.set_xlabel('Feature')
        ax.set_ylabel('Wasserstein Distance')

        plt.savefig(f'{self.output_folder}{sep}Wasserstein_distance_by_feature', bbox_inches="tight")
        plt.close()

    def plot_distance_by_feature_by_class(self):
        width = 0.4

        x = np.arange(len(self.wasserstein_distance_by_feature_by_class.keys()))

        class_0 = []
        class_1 = []
        for feature in sorted(self.wasserstein_distance_by_feature_by_class):
            for label in sorted(self.wasserstein_distance_by_feature_by_class[feature]):
                if label == 0:
                    class_0.append(self.wasserstein_distance_by_feature_by_class[feature][label])
                elif label == 1:
                    class_1.append(self.wasserstein_distance_by_feature_by_class[feature][label])
                else:
                    raise ValueError

        fig, ax = plt.subplots()

        fig.set_figwidth(60)
        fig.set_figheight(15)

        p1 = ax.bar(x, class_0, width, label='Class 0')
        p2 = ax.bar(x + width, class_1, width, label='Class 1')

        ax.set_xticks(x + width / 2)
        ax.set_xticklabels([f'f_{idx}' for idx in x])

        ax.bar_label(p1, padding=3, fmt='%.2f')
        ax.bar_label(p2, padding=3, fmt='%.2f')

        ax.set_xlabel('Feature')
        ax.set_ylabel('Wasserstein Distance')

        ax.set_title('Wasserstein Distance between Original and Generated images')

        plt.savefig(f'{self.output_folder}{sep}Wasserstein_distance_by_feature_and_class.png')
        plt.close()

    def plot_distances_between_classes_within_datasets(self, wasserstein_distances_between_classes_within_datasets):
        # feature: (dist_test, dist_generated)

        lines = [
            'Comparing the Distance between the activation distribution for class 0 and class 1 within one dataset (e.g. test set)\n',
            '\n',
            '-----------------\n',
        ]

        distances_by_feature_index = []

        for feature in sorted(wasserstein_distances_between_classes_within_datasets.keys()):
            dist_test, dist_gen = wasserstein_distances_between_classes_within_datasets[feature]

            lines.extend([
                f'Feature {feature}:\n',
                f'Distance in test set: {dist_test}\n',
                f'Distance in generated set: {dist_gen}\n',
                '-----------------\n',
            ])

            distances_by_feature_index.append((dist_test, dist_gen))

        test_distances = list(map(lambda d: d[0], distances_by_feature_index))
        gen_distances = list(map(lambda d: d[1], distances_by_feature_index))

        test_mean = np.mean(test_distances)
        test_std = np.std(test_distances)

        gen_mean = np.mean(gen_distances)
        gen_std = np.std(gen_distances)

        lines.extend([
            f'Mean distance in test set: {test_mean} -- avg distance in test set: {test_std}\n',
            f'Mean distance in generated set: {gen_mean} -- avg distance in generated set: {gen_std}\n',
        ])

        with open(f'{self.output_folder}{sep}wasserstein_distances_between_classes_within_dataset.txt', 'w') as file:
            file.writelines(lines)

        self._barplot_distances(test_distances, 'Test')
        self._barplot_distances(gen_distances, 'Generated')
        self._joint_barplot_distances(test_distances, gen_distances)
        self._barplot_distance_differences(test_distances, gen_distances)

    def _barplot_distances(self, distances, dataset_name):
        fig, ax = plt.subplots()

        fig.set_figwidth(40)
        fig.set_figheight(15)

        x = np.arange(len(distances))
        y = distances

        bar = ax.bar(x, y)
        ax.set_title(f'Wasserstein Distance between Class 0 and Class 1 Activation Distributions -- {dataset_name}')
        ax.bar_label(bar, padding=3, fmt='%.2f')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{idx}' for idx in x])

        ax.set_xlabel('Feature')
        ax.set_ylabel('Wasserstein Distance')

        plt.savefig(f'{self.output_folder}{sep}Wasserstein_distance_by_feature_within_dataset_{dataset_name}')
        plt.close()

    def _joint_barplot_distances(self, test_distances, gen_distances):
        fig, ax = plt.subplots()

        width = 0.4

        x = np.arange(len(test_distances))

        fig.set_figwidth(60)
        fig.set_figheight(15)

        p1 = ax.bar(x, test_distances, width, label='Test')
        p2 = ax.bar(x + width, gen_distances, width, label='Generated')

        ax.set_xticks(x + width / 2)
        ax.set_xticklabels([f'f_{idx}' for idx in x])

        ax.bar_label(p1, padding=3, fmt='%.2f')
        ax.bar_label(p2, padding=3, fmt='%.2f')

        ax.set_xlabel('Feature')
        ax.set_ylabel('Wasserstein Distance')

        ax.set_title('Wasserstein Distance between Class 0 and Class 1 Activation Distributions')

        plt.savefig(f'{self.output_folder}{sep}Wasserstein_distance_by_feature_within_dataset_comparison.png')
        plt.close()

    def _barplot_distance_differences(self, test_distances, gen_distances):
        differences = np.abs(np.array(test_distances) - np.array(gen_distances))

        fig, ax = plt.subplots()

        x = np.arange(len(differences))
        y = differences

        bar = ax.bar(x, y)
        ax.set_title(f'Differences in Wasserstein Distance between Class 0 and Class 1 Activation Distributions Test and Generated')
        ax.bar_label(bar, padding=3, fmt='%.2f')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{idx}' for idx in x])

        ax.set_xlabel('Feature')
        ax.set_ylabel('Difference in Wasserstein Distance')

        plt.savefig(f'{self.output_folder}{sep}Wasserstein_distance_difference_within_dataset')
        plt.close()

    def plot_distances_between_classes_between_datasets(self, wasserstein_distances_between_datasets):
        # shape: feature: orig0-orig1, alter0-orig1, alter1-orig0

        lines = [
            'Comparing the Distance between the activation distribution for two classes between test and generated set.\n',
            'This means comparing three distances per feature: Test set Class 0 vs. Test set Class 1, Generated set Class 0 vs. Test set Class 1, and Generated set Class 1 vs. Test set Class 0.\n',
            'Test set Class 0 vs. Test set Class 1 is the same as Test set Class 1 vs. Test set Class 0, since Wasserstein is a metric and therefore symmetric.\n',
            'The goal of the analysis is to figure out whether the generated images are closer to the other class.\n'
            '\n',
            '-----------------\n',
        ]

        test0_test1_dist = []
        gen0_test1_dist = []
        gen1_test0_dist = []

        for feature in sorted(wasserstein_distances_between_datasets.keys()):
            test0_test1, gen0_test1, gen1_test0 = wasserstein_distances_between_datasets[feature]

            lines.extend([
                f'Distance between Test set Class 0 and Class 1: {test0_test1}\n',
                f'Distance between Generated set Class 0 and Test set Class 1: {gen0_test1}\n',
                f'Distance between Generated set Class 1 and Test set Class 0: {gen1_test0}\n',
                '-----------------\n',
            ])

            test0_test1_dist.append(test0_test1)
            gen0_test1_dist.append(gen0_test1)
            gen1_test0_dist.append(gen1_test0)

        with open(f'{self.output_folder}{sep}wasserstein_distances_between_classes_between_dataset.txt', 'w') as file:
            file.writelines(lines)

        self._barplot_distances_between_datasets(test0_test1_dist, 'Test 0 - 1')
        self._barplot_distances_between_datasets(gen0_test1_dist, 'Generated 0 - Test 1')
        self._barplot_distances_between_datasets(gen1_test0_dist, 'Generated 1 - Test 0')

        self._joint_barplot_distances_between_datasets(test0_test1_dist, gen0_test1_dist, gen1_test0_dist)

    def _barplot_distances_between_datasets(self, distances, name):
        fig, ax = plt.subplots()

        fig.set_figwidth(40)
        fig.set_figheight(15)

        x = np.arange(len(distances))
        y = distances

        bar = ax.bar(x, y)
        ax.set_title(f'Wasserstein Distance between {name}')
        ax.bar_label(bar, padding=3, fmt='%.2f')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{idx}' for idx in x])

        ax.set_xlabel('Feature')
        ax.set_ylabel('Wasserstein Distance')

        plt.savefig(f'{self.output_folder}{sep}Wasserstein_distance_by_feature_between_dataset_{name}')
        plt.close()

    def _joint_barplot_distances_between_datasets(self, test0_test1_dist, gen0_test1_dist, gen1_test0_dist):
        fig, ax = plt.subplots()

        width = 0.3

        x = np.arange(len(test0_test1_dist))

        fig.set_figwidth(60)
        fig.set_figheight(15)

        p1 = ax.bar(x, test0_test1_dist, width, label='Test 0 Test 1')
        p2 = ax.bar(x + width, gen0_test1_dist, width, label='Generated 0 Test 1')
        p3 = ax.bar(x + 2*width, gen1_test0_dist, width, label='Generated 1 Test 0')

        ax.set_xticks(x + 2*width / 3)
        ax.set_xticklabels([f'f_{idx}' for idx in x])

        ax.bar_label(p1, padding=3, fmt='%.2f')
        ax.bar_label(p2, padding=3, fmt='%.2f')
        ax.bar_label(p3, padding=3, fmt='%.2f')

        ax.set_xlabel('Feature')
        ax.set_ylabel('Wasserstein Distance')

        ax.set_title('Wasserstein Distance between Activation Distributions')

        plt.savefig(f'{self.output_folder}{sep}Wasserstein_distance_by_feature_between_dataset_comparison.png')
        plt.close()

