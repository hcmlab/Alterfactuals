import re

import random

import torchvision
import os
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image

from main.countercounter.classifier.dataset.DataSet import ExecutionType
from main.countercounter.classifier.dataset.DataSetMNIST import DataSetMNIST

expected_size_by_execution_type = {
    ExecutionType.TRAIN: 5515,
    ExecutionType.VAL: 597,
    ExecutionType.TEST: 1010,
}


class DataSetMNIST1Class(DataSetMNIST):

    def __init__(self, execution_type: ExecutionType, root_dir: str = '', transform=None, in_memory: bool = True,
                 gray: bool = False, expected_size_by_execution_type=expected_size_by_execution_type, one_hot_labels=False):
        super().__init__(execution_type, root_dir, transform, in_memory, gray, expected_size_by_execution_type, one_hot_labels=one_hot_labels)

