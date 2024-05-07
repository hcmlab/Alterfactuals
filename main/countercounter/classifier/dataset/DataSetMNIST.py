import re

import random

import PIL
import torchvision
import os
import torch
from torch.utils.data import Dataset

from main.countercounter.classifier.dataset.DataSet import ExecutionType

TRAIN_DIR = '/TRAIN'
VAL_DIR = '/VAL'
TEST_DIR = '/TEST'


dir_by_execution_type = {
    ExecutionType.TRAIN: TRAIN_DIR,
    ExecutionType.VAL: VAL_DIR,
    ExecutionType.TEST: TEST_DIR,
}

expected_size_by_execution_type = {
    ExecutionType.TRAIN: 10784,
    ExecutionType.VAL: 1198,
    ExecutionType.TEST: 1984,
}

one_hot_encoded = {
    3: 0,
    8: 1,
}


class DataSetMNIST(Dataset):

    def __init__(
            self,
            execution_type: ExecutionType,
            root_dir: str='',
            transform=None,
            in_memory: bool=True,
            gray: bool=False,
            expected_size_by_execution_type=expected_size_by_execution_type,
            encoding=one_hot_encoded,
            one_hot_labels=False,
            size_check=True,
    ):
        self.execution_type = execution_type
        self.root_dir = root_dir
        self.transform = transform
        self.transform_composed = torchvision.transforms.Compose(transform) if transform else None
        self.in_memory = in_memory
        self.gray = gray

        self.expected_size_by_execution_type = expected_size_by_execution_type

        self.encoding = encoding
        self.one_hot_labels = one_hot_labels

        self._initialize()

        if size_check:
            self._assert_data_set_size()

    def _assert_data_set_size(self):
        assert len(self.filename_by_index.keys()) == self.expected_size_by_execution_type[self.execution_type]

    def __len__(self):
        return len(self.filename_by_index.keys())

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.in_memory:
            return self.data[idx]
        else:
            return self._load_image(self.filename_by_index[idx]), self.get_label(self.filename_by_index[idx])

    def _load_image(self, filename):
        # image = read_image(filename) / 255.
        image = PIL.Image.open(filename)

        if self.transform:
            image = self.transform_composed(image)

        return image

    def _initialize(self) -> None:
        random.seed(42)

        self.filename_by_index = {}
        self.data = []

        data_dir = self._data_dir
        for idx, filename in enumerate(os.listdir(data_dir)):
            complete_filename = os.path.join(data_dir, filename)

            self.filename_by_index[idx] = complete_filename

            if self.in_memory:
                self.data.append((self._load_image(complete_filename), self.get_label(complete_filename)))

    @property
    def _data_dir(self) -> str:
        return self.root_dir + dir_by_execution_type[self.execution_type]

    def get_label(self, filename: str):
        label_regex = 'label(\d)__'
        label_part = re.search(label_regex, filename)
        label = label_part.group(1)

        label = self.encoding[int(label)]

        if self.one_hot_labels:
            if label == 0:
                return torch.tensor([1, 0])
            else:
                return torch.tensor([0, 1])
        return label
