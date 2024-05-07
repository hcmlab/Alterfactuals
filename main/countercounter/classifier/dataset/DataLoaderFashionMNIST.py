import torchvision
from PIL import Image
from torch.utils.data import DataLoader

from main.countercounter.classifier.dataset.DataSetFashionMNIST import DataSetFashionMNIST
from main.countercounter.classifier.dataset.DataSetFashionMNISTFull import DataSetFashionMNISTFull
from main.countercounter.classifier.dataset.DataSetFashionMNISTIdentified import DataSetFashionMNISTIdentified
from main.countercounter.classifier.dataset.DataSetMNIST import DataSetMNIST, ExecutionType

mean = 0.5
std = 0.5


class DataFashionMNIST:

    def __init__(self):
        self.transforms = [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Resize(128, Image.NEAREST), # bilinear etc. changes value ranges
                torchvision.transforms.Normalize(mean=mean,
                                     std=std),
            ]

    def get_loader(
            self,
            execution_type: ExecutionType,
            batch_size: int,
            num_workers: int = 1,
            in_memory: bool = True,
            root_dir = None,
            full_set=False,
            one_hot_labels=False,
            add_identifier=False,
            size_check=True,
            shuffle=True,
    ) -> DataLoader:
        if full_set:
            return self._get_full_loader(batch_size, in_memory, num_workers, root_dir, one_hot_labels, shuffle)
        else:
            return self._get_partial_loader(batch_size, execution_type, in_memory, num_workers, root_dir, one_hot_labels, add_identifier, size_check, shuffle)

    def _get_partial_loader(self, batch_size, execution_type, in_memory, num_workers, root_dir, one_hot_labels, add_identifier, size_check, shuffle):

        if add_identifier:
            dataset = DataSetFashionMNISTIdentified
        else:
            dataset = DataSetFashionMNIST
        return DataLoader(
            dataset(
                execution_type=execution_type,
                in_memory=in_memory,
                root_dir=root_dir,
                transform=self.transforms,
                one_hot_labels=one_hot_labels,
                size_check=size_check,
            ),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )

    def _get_full_loader(self, batch_size, in_memory, num_workers, root_dir, one_hot_labels, shuffle):
        return DataLoader(
            DataSetFashionMNISTFull(
                in_memory=in_memory,
                root_dir=root_dir,
                transform=self.transforms,
                one_hot_labels=one_hot_labels,
            ),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )


def denormalize(data):
    return (data * std) + mean
