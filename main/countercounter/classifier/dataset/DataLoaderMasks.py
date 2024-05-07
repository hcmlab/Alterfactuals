import torchvision
from torch.utils.data import DataLoader

from main.countercounter.classifier.dataset.DataSetMNIST import ExecutionType
from main.countercounter.classifier.dataset.DataSetMasks import DataSetMasks
from main.countercounter.classifier.dataset.DataSetMasksIdentified import DataSetMasksIdentified

mean = 0.5
std = 0.5


class DataMasks:

    def get_loader(
            self,
            execution_type: ExecutionType,
            batch_size: int,
            num_workers: int = 1,
            in_memory: bool = True,
            root_dir = None,
            full_set=False, # TODO
            one_hot_labels=False,
            size_check=True,
            add_identifier=False,
            shuffle=True,
    ) -> DataLoader:
        transforms = [
            # torchvision.transforms.Resize(128, Image.NEAREST),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=mean,
                                             std=std),
        ]

        if add_identifier:
            dataset = DataSetMasksIdentified
        else:
            dataset = DataSetMasks
        return DataLoader(
            dataset(
                execution_type=execution_type,
                in_memory=in_memory,
                root_dir=root_dir,
                transform=transforms,
                one_hot_labels=one_hot_labels,
                size_check=size_check,
            ),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )


def denormalize(data):
    return (data * std) + mean
