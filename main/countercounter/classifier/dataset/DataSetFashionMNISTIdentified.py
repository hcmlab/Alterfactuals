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
    ExecutionType.TRAIN: 10800,
    ExecutionType.VAL: 1200,
    ExecutionType.TEST: 2000,
}

one_hot_encoded = {
    7: 0,
    9: 1,
}


class DataSetFashionMNISTIdentified(DataSetMNISTIdentified):

    def __init__(self, execution_type: ExecutionType, root_dir: str = '', transform=None, in_memory: bool = True,
                 gray: bool = False, expected_size_by_execution_type=expected_size_by_execution_type, encoding=one_hot_encoded,
                 one_hot_labels=False, size_check=True,):
        super().__init__(execution_type, root_dir, transform, in_memory, gray, expected_size_by_execution_type, encoding, one_hot_labels=one_hot_labels, size_check=size_check)
