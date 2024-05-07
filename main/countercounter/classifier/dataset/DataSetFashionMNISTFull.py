from main.countercounter.classifier.dataset.DataSetFashionMNIST import expected_size_by_execution_type, one_hot_encoded
from main.countercounter.classifier.dataset.DataSetMNISTFull import DataSetMNISTFull


class DataSetFashionMNISTFull(DataSetMNISTFull):

    def __init__(self, in_memory, root_dir, transform, one_hot_labels):
        super().__init__(
            root_dir=root_dir,
            transform=transform,
            in_memory=in_memory,
            expected_size_by_execution_type=expected_size_by_execution_type,
            encoding=one_hot_encoded,
            one_hot_labels=one_hot_labels,
        )

