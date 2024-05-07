import torch
from torch import nn

from main.countercounter.autoencoder.networks.AE_networks import AEBigNet, AEV2BiggerNet, \
    AEBiggerNet, AEBiggerNoTransposeNet, AESmallNoTransposeNet, AEDhurandhar
from main.countercounter.autoencoder.training.Setup import Setup
from main.countercounter.classifier.dataset.DataLoaderFashionMNIST import DataFashionMNIST
from main.countercounter.classifier.dataset.DataLoaderFashionMasks import DataMasks
from main.countercounter.classifier.dataset.DataLoaderMNIST import DataMNIST
from main.countercounter.classifier.dataset.DataSet import ExecutionType
from main.countercounter.gan.ssim.SSIMLossSimilarityEncouraged import SSIMLossSimilarityEncouraged
from main.countercounter.gan.utils.AbstractTraining import DEVICE


DIR = 'dir'

CHECKPOINTS = 'checkpoints'
DATASET = 'dataset'

WEIGHT = 'weight_init'
MEAN = 'mean'
STD = 'std'


class SetupConfigurator:

    def __init__(self, config, root_dir, config_nr, eval=False, path_to_dataset=None, add_identifier=False):
        self.root_dir = root_dir
        self.config_nr = config_nr
        self.config = config
        self.path_to_dataset = path_to_dataset

        self.eval = eval

        self.add_identifier = add_identifier

        self.setup = Setup()

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

        self._setup_losses()

        return self.setup

    def _setup_losses(self):
        if 'loss' not in self.config['models'] or self.config['models']['loss'] == 'mse':
            self.setup.criterion = nn.MSELoss()
        elif self.config['models']['loss'] == 'ssim':
            self.setup.criterion = SSIMLossSimilarityEncouraged(1, DEVICE, self.in_channels)
        else:
            raise ValueError('Unknown loss function')

    def _setup_models(self):
        model_config = self.config['models']
        type = model_config['type']
        grayscale = model_config['grayscale']

        self.in_channels = 1 if grayscale else 3

        if self._is_default(type):
            self.setup.model = AEDhurandhar(self.in_channels).to(DEVICE)
        elif type == 'defaultnotranspose':
            self.setup.model = AEV2BiggerNet(self.in_channels).to(DEVICE)
        elif type == 'big':
            self.setup.model = AEBigNet(self.in_channels).to(DEVICE)
        elif type == 'bigger':
            self.setup.model = AEBiggerNet(self.in_channels).to(DEVICE)
        elif type == 'biggernotranspose':
            self.setup.model = AEBiggerNoTransposeNet(self.in_channels).to(DEVICE)
        elif type == 'smallnotranspose':
            self.setup.model = AESmallNoTransposeNet(self.in_channels).to(DEVICE)
        else:
            raise ValueError(f'Unknown model type')

    def _is_default(self, type):
        return type == 'default'

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

        size_check = True
        if 'size_check' in dataset_config:
            size_check = dataset_config['size_check']

        if self.path_to_dataset is not None:
            dir = self.path_to_dataset
        else:
            dir = self._add_root_dir(self.config[DIR]['dataset'])

        one_hot_labels = False
        
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
