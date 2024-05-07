from os.path import sep

import torchvision.transforms

import torch
import pandas as pd
import os

from torchvision.utils import save_image

from main.countercounter.classifier.dataset.DataLoaderMNIST import denormalize
from main.countercounter.classifier.emotion_classifier.CustomNets import SoftmaxLogitWrapper
from main.countercounter.classifier.evaluation.DistributionCalculator import DistributionCalculator
from main.countercounter.gan.utils.AbstractTraining import DEVICE



class EvaluationPreparer:

    def __init__(
            self,
            setup,
            path_to_output_folder='',
    ):
        self.test_loader = setup.test_loader
        self.val_loader = setup.val_loader
        self.train_loader = setup.train_loader

        if not isinstance(setup.classifier.model, SoftmaxLogitWrapper):
            raise NotImplementedError
        self.classifier = setup.classifier

        self.path_to_output_folder = path_to_output_folder

        self.generator = setup.generator.to(DEVICE)
        self.raw_ssim = setup.ssim_function



    def evaluate(self):
        print(f'Evaluating:')
        print('---------------------------------------------------------')

        loaders = [
            (self.train_loader, 'TRAIN'),
            (self.val_loader, 'VAL'),
            (self.test_loader, 'TEST'),
        ]

        for loader, dir in loaders:
            self.test_data_alter = []
            self.generated_data_alter = []

            self.test_data_counter = []
            self.generated_data_counter = []

            self._store_data(loader, dir)

    def _store_data(self, loader, dir):
        alter_ssim = []
        counter_ssim = []

        count = 0

        full_path_images = f'{self.path_to_output_folder}{sep}{"Images"}{sep}{dir}'
        full_path_images_rgb = f'{self.path_to_output_folder}{sep}{"ImagesRGB"}{sep}{dir}'
        os.makedirs(full_path_images)
        os.makedirs(full_path_images_rgb)

        full_path_files = f'{self.path_to_output_folder}{sep}{dir}'
        os.mkdir(full_path_files)

        for imgs, labels, filename in loader:
            # if count > 10:
            #     break
            count += 1

            imgs = imgs.to(DEVICE)
            generated = self.generator(imgs, labels)

            pred_original = self.classifier.pred(imgs)
            label_original = torch.argmax(pred_original, 1)

            pred_generated = self.classifier.pred(generated)
            labels_generated = torch.argmax(pred_generated, 1)

            is_alterfactual = label_original == labels_generated
            gen_filename = self._gen_filename(filename[0], full_path_images, is_alterfactual)
            gen_filename_rgb = self._gen_filename(filename[0], full_path_images_rgb, is_alterfactual)

            ssim = self._get_ssim(imgs, generated)
            data_ssim = (filename[0], gen_filename, ssim)

            data_orig = (imgs.cpu().detach(), label_original.item(), filename)  # filename is a tuple, but activation calculator handles it already
            data_gen = (generated.cpu().detach(), labels_generated.item(), gen_filename)

            if is_alterfactual:
                alter_ssim.append(data_ssim)
                self.test_data_alter.append(data_orig)
                self.generated_data_alter.append(data_gen)
            else:  # counterfactual
                counter_ssim.append(data_ssim)
                self.test_data_counter.append(data_orig)
                self.generated_data_counter.append(data_gen)

            # store generated image
            image_denorm = denormalize(generated)
            torchvision.transforms.ToPILImage()(image_denorm.squeeze(1)).save(gen_filename) # save_image converts to 3 channels by default

            # store as RGB as well (FID needs RGB)
            save_image(image_denorm.repeat(1, 3, 1, 1), gen_filename_rgb)
            save_image(image_denorm, gen_filename_rgb)#


        if self.test_data_alter:
            self.df_test_alter, self.df_test_feature_count_alter, self.df_logit_test_alter = DistributionCalculator(self.test_data_alter, self.classifier.model.model).calculate_with_identifier()
            self.df_generated_alter, self.df_generated_feature_count_alter, self.df_logit_generated_alter = DistributionCalculator(self.generated_data_alter, self.classifier.model.model).calculate_with_identifier()

            self._store_df(self.df_test_alter, 'Alter Original Activations', full_path_files)
            self._store_df(self.df_logit_test_alter, 'Alter Original Logits', full_path_files)
            self._store_df(self.df_generated_alter, 'Alter Generated Activations', full_path_files)
            self._store_df(self.df_logit_generated_alter, 'Alter Generated Logits', full_path_files)

        if self.test_data_counter:
            self.df_test_counter, self.df_test_feature_count_counter, self.df_logit_test_counter = DistributionCalculator(self.test_data_counter, self.classifier.model.model).calculate_with_identifier()
            self.df_generated_counter, self.df_generated_feature_count_counter, self.df_logit_generated_counter = DistributionCalculator(self.generated_data_counter, self.classifier.model.model).calculate_with_identifier()

            self._store_df(self.df_test_counter, 'Counter Original Activations', full_path_files)
            self._store_df(self.df_logit_test_counter, 'Counter Original Logits', full_path_files)
            self._store_df(self.df_generated_counter, 'Counter Generated Activations', full_path_files)
            self._store_df(self.df_logit_generated_counter, 'Counter Generated Logits', full_path_files)

        self._make_and_store_ssim_dataframes(alter_ssim, counter_ssim, full_path_files)

    def _get_ssim(self, image1, image2):
        i1 = denormalize(image1)
        i2 = denormalize(image2)

        return self.raw_ssim(i1, i2).item()

    def _gen_filename(self, filename, full_path, is_alterfactual):
        splits = filename.split('.')

        assert len(splits) == 2

        old_path = splits[0]
        path_splits = old_path.split(sep)
        core_file = path_splits[-1]

        new_filename = full_path + sep + f'{"ALTER_" if is_alterfactual else "COUNTER_"}' + core_file + '_generated' + '.' + splits[1]
        return new_filename

    def _make_and_store_ssim_dataframes(self, alter_ssim, counter_ssim, full_path):
        data = [
            (alter_ssim, 'Alterfactuals'),
            (counter_ssim, 'Counterfactuals')
        ]

        for d, name in data:
            if d:
                self._make_and_store_ssim_dataframe(d, name, full_path)

    def _make_and_store_ssim_dataframe(self, data, name, full_path):
        # data has shape: (filename, gen_filename, ssim)
        columns = ['Filename1', 'Filename2', 'SSIM']
        df = pd.DataFrame(data, columns=columns)

        file_name = f'{full_path}{sep}{name}_ssim.csv'
        df.to_csv(file_name, sep=';')

    def _store_df(self, df, name, full_path):
        df.to_csv(f'{full_path}{sep}{name}.csv', sep=';')
