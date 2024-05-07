import random

import torch
from torch import nn

from main.countercounter.classifier.dataset.DataLoaderFashionMNIST import DataFashionMNIST
from main.countercounter.classifier.dataset.DataLoaderFashionMasks import DataMasks
from main.countercounter.classifier.dataset.DataLoaderMNIST import DataMNIST
from main.countercounter.classifier.dataset.DataLoaderRSNA import DataRSNA
from main.countercounter.classifier.dataset.DataSet import ExecutionType
from main.countercounter.classifier.emotion_classifier.CustomNets import CustomNet, CustomNetSmall, \
    CustomNetSmallBinary, CustomNetBinary, CustomNetVerySmall, CustomNetExtremelySmallBinary, CustomNetSmallLogits, \
    SoftmaxWrapper, CustomNetSmallGAPLogits, SoftmaxLogitWrapper, CustomNetSmallNoGAPLogits, CustomNetBigGAPLogits, \
    ModifiedAlexNet
from main.countercounter.classifier.emotion_classifier.ResnetWrapper import ResnetWrapper
from main.countercounter.classifier.training.Setup import Setup
from main.countercounter.gan.utils.AbstractTraining import DEVICE
from piqa.ssim import SSIM


DIR = 'dir'

CHECKPOINTS = 'checkpoints'
DATASET = 'dataset'

WEIGHT = 'weight_init'
MEAN = 'mean'
STD = 'std'


class SetupConfigurator:

    def __init__(self, config, root_dir, config_nr, eval=False, path_to_dataset=None, add_identifier=False, shuffle=True):
        self.root_dir = root_dir
        self.config_nr = config_nr
        self.config = config
        self.path_to_dataset = path_to_dataset

        self.eval = eval

        self.add_identifier = add_identifier

        self.setup = Setup()
        self.shuffle = shuffle

        if not shuffle:  # for deterministic data loaders
            random.seed(42)
            torch.manual_seed(42)

    def configure(self) -> Setup:
        self._setup_models()
        self.setup.optimizer = self._get_optimizer(self.setup.model)

        self.setup.train_loader = self._get_loader(ExecutionType.TRAIN)
        self.setup.val_loader = self._get_loader(ExecutionType.VAL)
        self.setup.test_loader = self._get_loader(ExecutionType.TEST)

        self.setup.epochs = self.config['epochs']

        self.setup.checkpoints_dir = self._add_root_dir(self.config[DIR]['checkpoints'])
        self.setup.root_dir = self.root_dir
        self.setup.config_nr = self.config_nr
        self.setup.tensorboard_dir = self._add_root_dir(self.config[DIR]['tensorboard'])
        self.setup.model_dir = self._add_root_dir(self.config[DIR]['model'])

        self.setup.ssim_function = self._ssim_function()

        self._setup_losses()

        return self.setup

    def _setup_losses(self):
        if isinstance(self.setup.model, CustomNet) or isinstance(self.setup.model, CustomNetSmall) or isinstance(self.setup.model, CustomNetVerySmall) or isinstance(self.setup.model, SoftmaxWrapper):
            self.setup.criterion = nn.CrossEntropyLoss()
        else:
            self.setup.criterion = nn.BCELoss()

    def _setup_models(self):
        model_config = self.config['models']
        type = model_config['type']
        grayscale = model_config['grayscale']
        n_classes = self.config[DATASET]['n_classes']

        if self._is_resnet50(type):
            self.setup.model = ResnetWrapper(n_classes=n_classes, size=50, grayscale=grayscale).to(DEVICE)
        elif self._is_resnet18(type):
            self.setup.model = ResnetWrapper(n_classes=n_classes, size=18, grayscale=grayscale).to(DEVICE)
        elif type == 'custom':
            self.setup.model = CustomNet(n_classes=n_classes).to(DEVICE)
        elif type == 'customsmall':
            self.setup.model = CustomNetSmall(n_classes=n_classes).to(DEVICE)
        elif type == 'customverysmall':
            self.setup.model = CustomNetVerySmall(n_classes=n_classes).to(DEVICE)
        elif type == 'customsmallbinary':
            self.setup.model = CustomNetSmallBinary().to(DEVICE)
        elif type == 'customextremelysmallbinary':
            self.setup.model = CustomNetExtremelySmallBinary().to(DEVICE)
        elif type == 'customsmallbinary3chan':
            self.setup.model = CustomNetSmallBinary(in_channels=3).to(DEVICE)
        elif type == 'custombinary3chan':
            self.setup.model = CustomNetBinary(in_channels=3).to(DEVICE)
        elif type == 'customsmalllogits':
            inner_model = CustomNetSmallLogits(n_classes=n_classes).to(DEVICE)
            self.setup.model = SoftmaxWrapper(inner_model).to(DEVICE)
        elif type == 'customsmallgaplogits':
            inner_model = CustomNetSmallGAPLogits(n_classes=n_classes).to(DEVICE)
            self.setup.model = SoftmaxLogitWrapper(inner_model).to(DEVICE)
        elif type == 'customsmallgaplogits3chan':
            inner_model = CustomNetSmallGAPLogits(n_classes=n_classes, in_channels=3).to(DEVICE)
            self.setup.model = SoftmaxLogitWrapper(inner_model).to(DEVICE)
        elif type == 'customsmallnogaplogits':
            inner_model = CustomNetSmallNoGAPLogits(n_classes=n_classes).to(DEVICE)
            self.setup.model = SoftmaxLogitWrapper(inner_model).to(DEVICE)
        elif type == 'custombiggaplogits':
            inner_model = CustomNetBigGAPLogits(n_classes=n_classes).to(DEVICE)
            self.setup.model = SoftmaxLogitWrapper(inner_model).to(DEVICE)
        elif type == 'custombiggaplogits3chan':
            inner_model = CustomNetBigGAPLogits(n_classes=n_classes, in_channels=3).to(DEVICE)
            self.setup.model = SoftmaxLogitWrapper(inner_model).to(DEVICE)
        elif type == 'modalexnext':
            inner_model = ModifiedAlexNet(n_classes=n_classes).to(DEVICE)
            self.setup.model = SoftmaxLogitWrapper(inner_model).to(DEVICE)
        elif type == 'modalexnext3chan':
            inner_model = ModifiedAlexNet(n_classes=n_classes, in_channels=3).to(DEVICE)
            self.setup.model = SoftmaxLogitWrapper(inner_model).to(DEVICE)
        else:
            raise ValueError(f'Unknown model type')

    def _is_resnet50(self, type):
        return type == 'Resnet50'

    def _is_resnet18(self, type):
        return type == 'Resnet18'

    def _get_optimizer(self, model):
        opt_config = self.config['optimizer']
        name = opt_config['name']
        lr = float(self.config['lr']['model'])

        if name == 'Adam':
            return torch.optim.Adam(model.parameters(), lr)

        return None

    def _get_loader(self, execution_type):
        dataset_config = self.config[DATASET]

        type = dataset_config['type']
        in_memory = dataset_config['in_memory']
        batch_size = self.config['batch_size']
        n_classes = dataset_config['n_classes']

        size_check = True
        if 'size_check' in dataset_config:
            size_check = dataset_config['size_check']

        if self.path_to_dataset is not None:
            dir = self.path_to_dataset
        else:
            dir = self._add_root_dir(self.config[DIR]['dataset'])

        one_hot_labels = isinstance(self.setup.model, CustomNet) or isinstance(self.setup.model, CustomNetSmall) or isinstance(self.setup.model, CustomNetVerySmall) or isinstance(self.setup.model, SoftmaxWrapper)
        
        if self._is_mnist(type):
            return DataMNIST().get_loader(
                execution_type=execution_type,
                batch_size=batch_size,
                num_workers=2,
                in_memory=in_memory,
                root_dir=dir,
                fake_class=False, # only use in GAN training
                one_hot_labels=one_hot_labels,
                add_identifier=self.add_identifier,
                size_check=size_check,
                shuffle=self.shuffle,
            )
        elif self._is_fashion_mnist(type):
            return DataFashionMNIST().get_loader(
                execution_type=execution_type,
                batch_size=batch_size,
                num_workers=2,
                in_memory=in_memory,
                root_dir=dir,
                one_hot_labels=one_hot_labels,
                add_identifier=self.add_identifier,
                size_check=size_check,
                shuffle=self.shuffle,
            )
        elif self._is_masks(type):
            return DataMasks().get_loader(
                execution_type=execution_type,
                batch_size=batch_size,
                num_workers=2,
                in_memory=in_memory,
                root_dir=dir,
                one_hot_labels=one_hot_labels,
                size_check=size_check,
                add_identifier=self.add_identifier,
                shuffle=self.shuffle,
            )
        elif self._is_rsna(type):
            return DataRSNA().get_loader(
                execution_type=execution_type,
                batch_size=batch_size,
                num_workers=2,
                in_memory=in_memory,
                root_dir=dir,
                one_hot_labels=one_hot_labels,
                add_identifier=self.add_identifier,
                size_check=size_check,
                shuffle=self.shuffle,
            )
        return None

    def _add_root_dir(self, path):
        return self.root_dir + '/' + path

    def _is_grayscale(self, type):
        return True

    def _is_mnist(self, type):
        return type == 'MNIST'

    def _is_fashion_mnist(self, type):
        return type == 'FashionMNIST'

    def _is_masks(self, type):
        return type == 'MASKS'

    def _is_rsna(self, type):
        return type == 'RSNA'

    def _ssim_function(self):
        model_config = self.config['models']
        type = model_config['type']

        n_channels = 1
        if '3chan' in type:
            n_channels = 3

        return SSIM(value_range=1, n_channels=n_channels, non_negative=True)