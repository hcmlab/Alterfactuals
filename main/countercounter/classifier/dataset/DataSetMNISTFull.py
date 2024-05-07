import os

import torch
import torchvision
import random

from main.countercounter.classifier.dataset.DataSetMNIST import DataSetMNIST, one_hot_encoded, expected_size_by_execution_type


class DataSetMNISTFull(DataSetMNIST):

    def __init__(self, in_memory, root_dir, transform, expected_size_by_execution_type=expected_size_by_execution_type,
            encoding=one_hot_encoded, one_hot_labels=False):
        super().__init__(None, root_dir, transform, in_memory, expected_size_by_execution_type=expected_size_by_execution_type,
            encoding=encoding, one_hot_labels=one_hot_labels)

        self.root_dir = root_dir
        self.transform = transform
        self.transform_composed = torchvision.transforms.Compose(transform) if transform else None
        self.in_memory = in_memory

    def _assert_data_set_size(self):
        assert len(self.filename_by_index.keys()) == sum(self.expected_size_by_execution_type.values())

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.in_memory:
            return self.data[idx]
        else:
            return self._load_image(self.filename_by_index[idx]), self.get_label(self.filename_by_index[idx]), self.partial_filename_by_index[idx], self.dirname_by_index[idx]

    @property
    def _data_dir(self) -> str:
        return self.root_dir

    def _initialize(self) -> None:
        random.seed(42)

        self.filename_by_index = {}
        self.partial_filename_by_index = {}
        self.dirname_by_index = {}

        self.data = []

        data_dir = self._data_dir

        counter = 0
        for dir_name in os.listdir(data_dir):
            dir_path = os.path.join(data_dir, dir_name)

            for filename in os.listdir(dir_path):
                complete_filename = os.path.join(dir_path, filename)

                self.filename_by_index[counter] = complete_filename
                self.partial_filename_by_index[counter] = filename
                self.dirname_by_index[counter] = dir_name

                counter += 1

                if self.in_memory:
                    self.data.append((self._load_image(complete_filename), self.get_label(complete_filename), filename, dir_name))