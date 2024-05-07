import itertools
import random
from collections import defaultdict
from os.path import sep

import torch
import pandas as pd

from main.countercounter.classifier.dataset.DataLoaderMNIST import denormalize
from main.countercounter.classifier.emotion_classifier.CustomNets import SoftmaxLogitWrapper, CustomNetSmallGAPLogits
from main.countercounter.classifier.evaluation.DistributionCalculator import DistributionCalculator


class TrainDistributionEvaluator:

    def __init__(self, setup, path_to_output_folder):
        self.train_loader = setup.train_loader

        self.raw_ssim = setup.ssim_function

        if not isinstance(setup.model, SoftmaxLogitWrapper):
            raise NotImplementedError

        self.softmax_model = setup.model

        self.path_to_output_folder = path_to_output_folder

    def evaluate(self):
        self.softmax_model.eval()
        ssim_by_test_image_pair_all = self._find_test_pairs()
        self._store_all_ssim_info(ssim_by_test_image_pair_all)

    def _find_test_pairs(self):
        ssim_by_test_image_pair_all = defaultdict(dict)
        
        outer = 0

        # do not iterate over data loader, takes wayyy to long
        image_filename_label_by_index = self._get_data(self.train_loader)
        idx = image_filename_label_by_index.keys()

        # cartesian product of all images
        all_pairs = itertools.product(idx, idx)

        for (idx1, idx2) in all_pairs:
            outer += 1
            if outer % 10000 == 0:
                print(f'Outer loop: {outer}')

            image1, filename1, label1 = image_filename_label_by_index[idx1]
            image2, filename2, label2 = image_filename_label_by_index[idx2]

            pair = self._make_pair(filename1, filename2)

            ssim = self._get_ssim(image1, image2)

            if pair[0] not in ssim_by_test_image_pair_all.keys() or pair[1] not in ssim_by_test_image_pair_all[
                pair[0]].keys():
                ssim_by_test_image_pair_all[pair[0]][pair[1]] = ssim
                ssim_by_test_image_pair_all[pair[1]][pair[0]] = ssim

        print('>>> Pair calculation finished')
        return ssim_by_test_image_pair_all

    def _make_pair(self, filename1, filename2):
        if filename1 < filename2:
            return (filename1, filename2)
        else:
            return (filename2, filename1)

    def _get_ssim(self, image1, image2):
        i1 = denormalize(image1)
        i2 = denormalize(image2)

        return self.raw_ssim(i1, i2).item()

    def _get_data(self, loader):
        image_filename_label_by_index = {}

        for idx, (image1, _, filename1) in enumerate(loader):
            filename1 = filename1[0]

            with torch.no_grad():
                preds = self.softmax_model(image1)

            _, label = torch.max(preds, dim=1)

            image_filename_label_by_index[idx] = (image1, filename1, label)

        return image_filename_label_by_index

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
        df.to_csv(f'{self.path_to_output_folder}{sep}train_ssim.csv', sep=';')