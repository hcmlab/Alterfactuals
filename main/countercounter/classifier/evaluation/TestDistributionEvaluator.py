import itertools
import random
from collections import defaultdict
from os.path import sep

import torch
import pandas as pd

from main.countercounter.classifier.dataset.DataLoaderMNIST import denormalize
from main.countercounter.classifier.emotion_classifier.CustomNets import SoftmaxLogitWrapper, CustomNetSmallGAPLogits
from main.countercounter.classifier.evaluation.DistributionCalculator import DistributionCalculator


class TestDistributionEvaluator:

    def __init__(self, setup, path_to_output_folder):
        self.test_loader = setup.test_loader

        self.raw_ssim = setup.ssim_function

        if not isinstance(setup.model, SoftmaxLogitWrapper):
            raise NotImplementedError
        
        self.softmax_model = setup.model
        
        self.model: CustomNetSmallGAPLogits = setup.model.model
        self.model.eval()

        self.path_to_output_folder = path_to_output_folder

    def evaluate(self):
        df_activations, feature_count, df_logits = DistributionCalculator(self.test_loader, self.model).calculate_with_identifier()
        # df_activations shape: Class, feature1, ..., featureN, filename
        # df_logits: Class, Logit_0, Logit_1, filename

        ssim_by_test_image_pair_same_labels, test_image_and_random_image_same_labels, ssim_by_test_image_pair_diff_labels, test_image_and_random_image_diff_labels, ssim_by_test_image_pair_all = self._find_test_pairs()
        max_min_ssim_info_by_filename_same_labels = self._get_max_min(ssim_by_test_image_pair_same_labels)
        max_min_ssim_info_by_filename_diff_labels = self._get_max_min(ssim_by_test_image_pair_diff_labels)

        df_act_pair_random, df_logit_pair_random = self._make_random_pair_df(df_activations, df_logits, test_image_and_random_image_same_labels)
        df_act_pair_max, df_logit_pair_max = self._make_max_or_min_pair(df_activations, df_logits, max_min_ssim_info_by_filename_same_labels, max=True)
        df_act_pair_min, df_logit_pair_min = self._make_max_or_min_pair(df_activations, df_logits, max_min_ssim_info_by_filename_same_labels, max=False)
        max_min_ssim_info_df_act = self._make_info_df(max_min_ssim_info_by_filename_same_labels)

        self._save_pair(df_act_pair_random, 'random_act')
        self._save_pair(df_act_pair_max, 'max_ssim_act')
        self._save_pair(df_act_pair_min, 'min_ssim_act')

        self._save_pair(df_logit_pair_random, 'random_logit')
        self._save_pair(df_logit_pair_max, 'max_ssim_logit')
        self._save_pair(df_logit_pair_min, 'min_ssim_logit')

        max_min_ssim_info_df_act.to_csv(f'{self.path_to_output_folder}{sep}max_min_ssim_info.csv', sep=';')
        self._store_info_on_random_pairs(test_image_and_random_image_same_labels, ssim_by_test_image_pair_same_labels)
        self._store_max_ssim_info(max_min_ssim_info_df_act, max=True)
        self._store_max_ssim_info(max_min_ssim_info_df_act, max=False)

        # and again, but this time over all pairs with different labels for counterfactuals
        df_act_pair_random, df_logit_pair_random = self._make_random_pair_df(df_activations, df_logits, test_image_and_random_image_diff_labels)
        df_act_pair_max, df_logit_pair_max = self._make_max_or_min_pair(df_activations, df_logits, max_min_ssim_info_by_filename_diff_labels, max=True)
        df_act_pair_min, df_logit_pair_min = self._make_max_or_min_pair(df_activations, df_logits, max_min_ssim_info_by_filename_diff_labels, max=False)
        max_min_ssim_info_df_act = self._make_info_df(max_min_ssim_info_by_filename_diff_labels)

        self._save_pair(df_act_pair_random, 'random_act_classes_combined')
        self._save_pair(df_act_pair_max, 'max_ssim_act_classes_combined')
        self._save_pair(df_act_pair_min, 'min_ssim_act_classes_combined')

        self._save_pair(df_logit_pair_random, 'random_logit_classes_combined')
        self._save_pair(df_logit_pair_max, 'max_ssim_logit_classes_combined')
        self._save_pair(df_logit_pair_min, 'min_ssim_logit_classes_combined')

        max_min_ssim_info_df_act.to_csv(f'{self.path_to_output_folder}{sep}max_min_ssim_info_classes_combined.csv', sep=';')
        self._store_info_on_random_pairs(test_image_and_random_image_diff_labels, ssim_by_test_image_pair_diff_labels, extension='_classes_combined')
        self._store_max_ssim_info(max_min_ssim_info_df_act, max=True, extension='_classes_combined')
        self._store_max_ssim_info(max_min_ssim_info_df_act, max=False, extension='_classes_combined')

        self._store_all_ssim_info(ssim_by_test_image_pair_all)

    def _find_test_pairs(self):
        ssim_by_test_image_pair_same_labels = defaultdict(dict)
        test_image_and_random_image_same_labels = []

        ssim_by_test_image_pair_diff_labels = defaultdict(dict)
        test_image_and_random_image_diff_labels = []

        ssim_by_test_image_pair_all = defaultdict(dict)
        
        outer = 0

        # do not iterate over data loader, takes wayyy to long
        image_filename_label_by_index = self._get_data(self.test_loader)
        idx = image_filename_label_by_index.keys()

        # cartesian product of all images
        all_pairs = itertools.product(idx, idx)

        for (idx1, idx2) in all_pairs:
            outer += 1
            if outer % 100000 == 0:
                print(f'Outer loop: {outer}')

            image1, filename1, label1 = image_filename_label_by_index[idx1]
            image2, filename2, label2 = image_filename_label_by_index[idx2]

            pair = self._make_pair(filename1, filename2)

            ssim = self._get_ssim(image1, image2)

            if pair[0] not in ssim_by_test_image_pair_all.keys() or pair[1] not in ssim_by_test_image_pair_all[
                pair[0]].keys():
                ssim_by_test_image_pair_all[pair[0]][pair[1]] = ssim
                ssim_by_test_image_pair_all[pair[1]][pair[0]] = ssim

            # Step 0a: skip if same image
            if idx1 == idx2:
                continue

            # Step 0b: if not same classifier prediction, skip as well
            if label1 != label2:
                if pair[0] not in ssim_by_test_image_pair_diff_labels.keys() or pair[1] not in \
                        ssim_by_test_image_pair_diff_labels[pair[0]].keys():
                    ssim_by_test_image_pair_diff_labels[pair[0]][pair[1]] = ssim
                    ssim_by_test_image_pair_diff_labels[pair[1]][pair[0]] = ssim
            else:
                # Step 1: if not already calculated, calculate SSIM and store
                if pair[0] not in ssim_by_test_image_pair_same_labels.keys() or pair[1] not in \
                        ssim_by_test_image_pair_same_labels[pair[0]].keys():
                    ssim_by_test_image_pair_same_labels[pair[0]][pair[1]] = ssim
                    ssim_by_test_image_pair_same_labels[pair[1]][pair[0]] = ssim

        print('cartesian product done')

        available_random_images_class_0 = set()
        available_random_images_class_1 = set()

        imag_file_labels = image_filename_label_by_index.values()

        for image, filename, label in imag_file_labels:
            if label.item() == 0:
                available_random_images_class_0.add(filename)
            else:
                available_random_images_class_1.add(filename)

        for filename1 in ssim_by_test_image_pair_same_labels.keys():
            if filename1 in available_random_images_class_0:
                other_filenames = available_random_images_class_0
            else:
                other_filenames = available_random_images_class_1
            filename2 = random.choice(list(other_filenames))
            other_filenames.remove(filename2)
            test_image_and_random_image_same_labels.append((filename1, filename2))

        for filename1 in ssim_by_test_image_pair_diff_labels.keys():
            if filename1 in available_random_images_class_0:
                other_filenames = available_random_images_class_1
            else:
                other_filenames = available_random_images_class_0
            filename2 = random.choice(list(other_filenames))
            other_filenames.remove(filename2)
            test_image_and_random_image_diff_labels.append((filename1, filename2))

        print('>>> Pair calculation finished')
        return ssim_by_test_image_pair_same_labels, test_image_and_random_image_same_labels, ssim_by_test_image_pair_diff_labels, test_image_and_random_image_diff_labels, ssim_by_test_image_pair_all

    def _make_pair(self, filename1, filename2):
        if filename1 < filename2:
            return (filename1, filename2)
        else:
            return (filename2, filename1)

    def _get_ssim(self, image1, image2):
        i1 = denormalize(image1)
        i2 = denormalize(image2)

        return self.raw_ssim(i1, i2).item()

    def _make_random_pair_df(self, df_act, df_logit, test_image_and_random_image):
        X_columns = sorted([col for col in df_act if col.startswith('feature')])
        y_column = ['Class']

        X_columns_logit = ['Logit_0', 'Logit_1']

        data1 = []
        data2 = []
        data3 = []
        data4 = []

        for filename1, filename2 in test_image_and_random_image:
            row1 = df_act.loc[df_act.Filename == filename1, X_columns + y_column].values.flatten().tolist()
            row2 = df_act.loc[df_act.Filename == filename2, X_columns + y_column].values.flatten().tolist()

            data1.append(row1)
            data2.append(row2)

            row3 = df_logit.loc[df_logit.Filename == filename1, X_columns_logit + y_column].values.flatten().tolist()
            row4 = df_logit.loc[df_logit.Filename == filename2, X_columns_logit + y_column].values.flatten().tolist()

            data3.append(row3)
            data4.append(row4)

        columns = X_columns + y_column

        df1 = pd.DataFrame(data1, columns=columns)
        df2 = pd.DataFrame(data2, columns=columns)

        columns = X_columns_logit + y_column

        df3 = pd.DataFrame(data3, columns=columns)
        df4 = pd.DataFrame(data4, columns=columns)

        return (df1, df2), (df3, df4)

    def _get_max_min(self, ssim_by_test_image_pair):
        # ssim_by_test_image_pair shape: f1: {f2: ssim}

        # target shape: [f, max ssim f, max ssim, min ssim f, min ssim] for all f (duplicates!)
        data = {}
        
        for f1 in ssim_by_test_image_pair.keys():
            min_file, min_ssim, max_file, max_ssim = self._find_min_max(ssim_by_test_image_pair[f1])
            data[f1] = [f1, min_file, min_ssim, max_file, max_ssim]
        
        return data

    def _find_min_max(self, ssim_by_filename):
        min_filename_min_ssim = min(ssim_by_filename.items(), key=lambda x: x[1])
        max_filename_max_ssim = max(ssim_by_filename.items(), key=lambda x: x[1])

        return (*min_filename_min_ssim, *max_filename_max_ssim)

    def _make_max_or_min_pair(self, df_act, df_logit, max_min_ssim_info_by_filename, max=True):
        X_columns = sorted([col for col in df_act if col.startswith('feature')])
        y_column = ['Class']

        X_columns_logits = ['Logit_0', 'Logit_1']

        data1 = []
        data2 = []
        data3 = []
        data4 = []

        for f1 in max_min_ssim_info_by_filename.keys():
            if max:  # shape: [f1, min_file, min_ssim, max_file, max_ssim]
                idx = 3
            else:
                idx = 1

            f2 = max_min_ssim_info_by_filename[f1][idx]

            row1 = df_act.loc[df_act.Filename == f1, X_columns + y_column].values.flatten().tolist()
            row2 = df_act.loc[df_act.Filename == f2, X_columns + y_column].values.flatten().tolist()

            data1.append(row1)
            data2.append(row2)

            row3 = df_logit.loc[df_logit.Filename == f1, X_columns_logits + y_column].values.flatten().tolist()
            row4 = df_logit.loc[df_logit.Filename == f2, X_columns_logits + y_column].values.flatten().tolist()

            data3.append(row3)
            data4.append(row4)

        columns = X_columns + y_column

        df1 = pd.DataFrame(data1, columns=columns)
        df2 = pd.DataFrame(data2, columns=columns)

        columns = X_columns_logits + y_column

        df3 = pd.DataFrame(data3, columns=columns)
        df4 = pd.DataFrame(data4, columns=columns)

        return (df1, df2), (df3, df4)

    def _make_info_df(self, max_min_ssim_info_by_filename):
        data = []

        for f in max_min_ssim_info_by_filename.keys():
            data.append(max_min_ssim_info_by_filename[f])

        columns = ['Filename', 'Min SSIM Filename', 'Min SSIM', 'Max SSIM Filename', 'Max SSIM']
        df = pd.DataFrame(data, columns=columns)
        return df

    def _save_pair(self, df_pair, name):
        df1, df2 = df_pair

        df1.to_csv(f'{self.path_to_output_folder}{sep}{name}_1.csv', sep=';')
        df2.to_csv(f'{self.path_to_output_folder}{sep}{name}_2.csv', sep=';')

    def _get_data(self, loader):
        image_filename_label_by_index = {}

        for idx, (image1, _, filename1) in enumerate(loader):
            filename1 = filename1[0]

            with torch.no_grad():
                preds = self.softmax_model(image1)

            _, label = torch.max(preds, dim=1)

            image_filename_label_by_index[idx] = (image1, filename1, label)

        return image_filename_label_by_index

    def _store_info_on_random_pairs(self, test_image_and_random_image, ssim_by_test_image_pair, extension=''):
        columns = ['Filename1', 'Filename2', 'SSIM']

        data = []
        for filename1, filename2 in test_image_and_random_image:
            data.append([filename1, filename2, ssim_by_test_image_pair[filename1][filename2]])

        df = pd.DataFrame(data, columns=columns)
        df.to_csv(f'{self.path_to_output_folder}{sep}info_on_random_pairs{extension}.csv', sep=';')

    def _store_max_ssim_info(self, df, max, extension=''):
        # structure so far: ['Filename', 'Min SSIM Filename', 'Min SSIM', 'Max SSIM Filename', 'Max SSIM']
        new_columns = ['Filename1', 'Filename2', 'SSIM']  # new target structure

        type = 'Max' if max else 'Min'

        ssim_col = f'{type} SSIM'
        name_col = f'{type} SSIM Filename'

        old_columns = ['Filename', name_col, ssim_col]

        data = df[old_columns].values.tolist()

        df = pd.DataFrame(data, columns=new_columns)
        df.to_csv(f'{self.path_to_output_folder}{sep}{type}_ssim_info{extension}.csv', sep=';')

    def _store_all_ssim_info(self, ssim_by_test_image_pair_all):
        data = []
        n = len(ssim_by_test_image_pair_all.keys())

        for f1 in sorted(ssim_by_test_image_pair_all.keys()):
            f1_data = [f1]
            for f2 in sorted(ssim_by_test_image_pair_all[f1].keys()):
                ssim = ssim_by_test_image_pair_all[f1][f2]
                ssim_inverted = 1 - ssim  # to be used as metric where 0 is no diff and 1 is max diff

                assert ssim_inverted >= 0
                assert ssim_inverted <= 1

                f1_data.append(ssim_inverted)

            assert len(f1_data) == n + 1   # + 1 to sort by this column later
            data.append(f1_data)

        assert len(data) == n

        columns = ['File'] + list(sorted(ssim_by_test_image_pair_all.keys()))

        df = pd.DataFrame(data, columns=columns)
        df.to_csv(f'{self.path_to_output_folder}{sep}all_ssim.csv', sep=';')