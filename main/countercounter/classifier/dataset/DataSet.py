from enum import Enum

import random

import torchvision
import os
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image


TRAIN_PERCENTAGE = 0.75
VAL_PERCENTAGE = 0.1
# Test percentage 0.15


TRAIN_DIR = '/TRAIN'
VAL_DIR = '/VAL'
TEST_DIR = '/TEST'


class ExecutionType(Enum):
    TRAIN = 0
    TEST = 1
    VAL = 2


dir_by_execution_type = {
    ExecutionType.TRAIN: TRAIN_DIR,
    ExecutionType.VAL: VAL_DIR,
    ExecutionType.TEST: TEST_DIR,
}

expected_size_by_execution_type = {
    ExecutionType.TRAIN: 1200,
    ExecutionType.VAL: 168,
    ExecutionType.TEST: 240,
}


def one_hot(idx_of_one: int):
    t = torch.zeros(8)
    t[idx_of_one] = 1.
    return t


emotions = {
    'angry': 0,
    'contemptuous': 1,
    'disgusted': 2,
    'fearful': 3,
    'happy': 4,
    'neutral': 5,
    'sad': 6,
    'surprised': 7,
}


class DataSet(Dataset):

    def __init__(
            self,
            execution_type: ExecutionType,
            root_dir: str=None,
            transform=None,
            in_memory: bool=True,
            gray: bool=False,
            expected_size_by_execution_type=expected_size_by_execution_type,
            emotions=emotions,
    ):
        self.execution_type = execution_type
        self.root_dir = root_dir
        self.transform = transform
        self.transform_composed = torchvision.transforms.Compose(transform) if transform else None
        self.in_memory = in_memory
        self.gray = gray

        self.expected_size_by_execution_type = expected_size_by_execution_type
        self.emotions = emotions

        self._initialize()

        assert len(self.filename_by_index.keys()) == self.expected_size_by_execution_type[self.execution_type]

    def __len__(self):
        return len(self.filename_by_index.keys())

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.in_memory:
            return self.data[idx]
        else:
            return self._load_image(self.filename_by_index[idx]), self.make_label(self._get_emotion(self.filename_by_index[idx]))

    def _load_image(self, filename):
        image = read_image(filename) / 255.

        if self.transform:
            image = self.transform_composed(image)

        return image

    def _initialize(self) -> None:
        random.seed(42)

        self.filename_by_index = {}
        self.idx_by_identity = {}

        self.data = []

        data_dir = self._data_dir
        for idx, filename in enumerate(os.listdir(data_dir)):
            complete_filename = os.path.join(data_dir, filename)

            self.filename_by_index[idx] = complete_filename

            emotion = self._get_emotion(complete_filename)
            if self.in_memory:
                self.data.append((self._load_image(complete_filename), self.make_label(emotion)))

            identity = filename.split('_')[1]
            if identity not in self.idx_by_identity.keys():
                self.idx_by_identity[identity] = []
            self.idx_by_identity[identity].append(idx)

    @property
    def _data_dir(self) -> str:
        return self.root_dir + dir_by_execution_type[self.execution_type]

    def make_label(self, emotion: str):
        return self.emotions[emotion]

    def _get_emotion(self, filename) -> str:
        return filename.split('_')[-3]