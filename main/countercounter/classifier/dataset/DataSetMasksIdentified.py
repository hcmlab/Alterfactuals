import random

import os

from main.countercounter.classifier.dataset.DataSet import ExecutionType
from main.countercounter.classifier.dataset.DataSetMNISTIdentified import DataSetMNISTIdentified

TRAIN_DIR = '/TRAIN'
VAL_DIR = '/VAL'
TEST_DIR = '/TEST'


dir_by_execution_type = {
    ExecutionType.TRAIN: TRAIN_DIR,
    ExecutionType.VAL: VAL_DIR,
    ExecutionType.TEST: TEST_DIR,
}

expected_size_by_execution_type = {
    ExecutionType.TRAIN: 89046,
    ExecutionType.VAL: 11130,
    ExecutionType.TEST: 11130,
}

one_hot_encoded = {
    0: 0,
    1: 1,
}


class DataSetMasksIdentified(DataSetMNISTIdentified):

    def __init__(self, execution_type: ExecutionType, root_dir: str = '', transform=None, in_memory: bool = True,
                 gray: bool = False, expected_size_by_execution_type=expected_size_by_execution_type, encoding=one_hot_encoded, one_hot_labels=False, size_check=True,):
        super().__init__(execution_type, root_dir, transform, in_memory, gray, expected_size_by_execution_type, encoding, one_hot_labels=one_hot_labels, size_check=size_check)

    def _initialize(self) -> None:
        random.seed(42)

        self.filename_by_index = {}
        self.data = []

        data_dir = self._data_dir

        counter = 0
        for sub_dir in os.listdir(data_dir):
            subdir_path = os.path.join(data_dir, sub_dir)
            for filename in os.listdir(subdir_path):
                complete_filename = os.path.join(subdir_path, filename)

                self.filename_by_index[counter] = complete_filename
                counter += 1

                if self.in_memory:
                    self.data.append((self._load_image(complete_filename), self.get_label(complete_filename), self.filename_by_index[counter]))