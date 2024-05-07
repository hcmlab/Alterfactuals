from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR

from main.countercounter.classifier.dataset.DataLoaderFashionMNIST import DataFashionMNIST
from main.countercounter.classifier.dataset.DataLoaderFashionMasks import DataMasks
from main.countercounter.classifier.dataset.DataLoaderMNIST import DataMNIST
from main.countercounter.classifier.dataset.DataSet import ExecutionType
from main.countercounter.csvae.networks.CSAE import CSAE
from main.countercounter.csvae.networks.CSVAE import CSVAE
from main.countercounter.csvae.networks.models import EncoderXToZ, EncoderXYToW, DecoderZWToX, DecoderZToY, EncoderYToW, \
    EncoderXYToWLessClass, EncoderXYToWEvenLessClass, EncoderXYToWNoClass, DecoderZWToXOther
from main.countercounter.csvae.networks.no_variation_models import EncoderXToZNoVar, EncoderXYToWLessClassNoVar, \
    EncoderXYToWEvenLessClassNoVar, EncoderXYToWNoClassNoVar, EncoderXYToWNoVar, EncoderYToWNoVar, DecoderZWToXNoVar, \
    DecoderZWToXOtherNoVar, DecoderZToYNoVar, EncoderXToZNoVarConv, EncoderXYToWNoVarConv, EncoderYToWNoVarConv, \
    DecoderZWToXOtherNoVarConv, DecoderZToYNoVarConv
from main.countercounter.gan.classifier.Classifier import Classifier
from main.countercounter.gan.ssim.SSIMLossSimilarityEncouraged import SSIMLossSimilarityEncouraged
from main.countercounter.gan.utils.AbstractTraining import DEVICE
from main.countercounter.csvae.training.Setup import Setup

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
        self._setup_loss_parameters()
        self._setup_models()
        self._setup_optimizers()

        self.setup.train_loader = self._get_loader(ExecutionType.TRAIN)
        self.setup.val_loader = self._get_loader(ExecutionType.VAL)
        self.setup.test_loader = self._get_loader(ExecutionType.TEST)

        self.setup.epochs = self.config['epochs']

        self.setup.checkpoints_dir = self._add_root_dir(self.config[DIR]['checkpoints'])
        self.setup.root_dir = self.root_dir
        self.setup.config_nr = self.config_nr
        self.setup.tensorboard_dir = self._add_root_dir(self.config[DIR]['tensorboard'])
        self.setup.model_dir = self._add_root_dir(self.config[DIR]['model'])
        self.setup.image_dir = self._add_root_dir(self.config[DIR]['image'])

        self.setup.image_sample_size = self.config['image_sample_size']

        return self.setup

    def _setup_models(self):
        model_config = self.config['models']
        type = model_config['type']
        grayscale = model_config['grayscale']

        self.in_channels = 1 if grayscale else 3
        input_size = 128
        n_classes = 1   # technically two, but this makes everything here a lot easier
        z_size = model_config['z_size']
        self.setup.z_size = z_size
        w_size = model_config['w_size']

        if 'no_var' in model_config and model_config['no_var']:
            self._setup_no_variation_models(input_size, model_config, n_classes, w_size, z_size)
        else:
            self._setup_variational_models(input_size, model_config, n_classes, w_size, z_size)

        classifier_config = self.config['classifier']

        name = classifier_config['model']
        dir = classifier_config['dir']
        size = classifier_config['size']
        n_classes = 2

        path = self._add_root_dir(dir) + '/' + f'{name}'

        loss = None

        in_channels = self.config[DATASET]['channels']

        self.setup.classifier = Classifier(path, DEVICE, size, loss, n_classes, in_channels)

    def _setup_variational_models(self, input_size, model_config, n_classes, w_size, z_size):
        self.setup.encoder_x_to_z = EncoderXToZ(input_size, z_size).to(DEVICE)
        if 'xywsplit' in model_config:
            if model_config['xywsplit'] == 1:
                self.setup.encoder_xy_to_w = EncoderXYToWLessClass(input_size, n_classes, w_size).to(DEVICE)
            elif model_config['xywsplit'] == 2:
                self.setup.encoder_xy_to_w = EncoderXYToWEvenLessClass(input_size, n_classes, w_size).to(DEVICE)
            elif model_config['xywsplit'] == 3:
                self.setup.encoder_xy_to_w = EncoderXYToWNoClass(input_size, n_classes, w_size).to(DEVICE)
            else:
                raise ValueError(f'Unknown xy w encoder split: {model_config["xywsplit"]}')
        else:
            self.setup.encoder_xy_to_w = EncoderXYToW(input_size, n_classes, w_size).to(DEVICE)
        self.setup.encoder_y_to_w = EncoderYToW(n_classes, w_size).to(DEVICE)
        if 'zwx' in model_config:
            if model_config['zwx'] == 1:
                self.setup.decoder_zw_to_x = DecoderZWToX(z_size, w_size, self.in_channels).to(DEVICE)
            if model_config['zwx'] == 2:
                self.setup.decoder_zw_to_x = DecoderZWToXOther(z_size, w_size, self.in_channels).to(DEVICE)
        else:
            self.setup.decoder_zw_to_x = DecoderZWToX(z_size, w_size, self.in_channels).to(DEVICE)
        self.setup.decoder_z_to_y = DecoderZToY(z_size, 2).to(DEVICE)
        self.setup.csvae = CSVAE(self.setup, DEVICE)

    def _setup_no_variation_models(self, input_size, model_config, n_classes, w_size, z_size):
        conv = 'conv' in model_config and model_config['conv']

        if conv:
            self.setup.encoder_x_to_z = EncoderXToZNoVarConv(input_size, z_size).to(DEVICE)
        else:
            self.setup.encoder_x_to_z = EncoderXToZNoVar(input_size, z_size).to(DEVICE)

        if 'xywsplit' in model_config:
            if model_config['xywsplit'] == 0:
                if conv:
                    self.setup.encoder_xy_to_w = EncoderXYToWNoVarConv(input_size, n_classes, w_size).to(DEVICE)
                else:
                    self.setup.encoder_xy_to_w = EncoderXYToWNoVar(input_size, n_classes, w_size).to(DEVICE)
            elif model_config['xywsplit'] == 1:
                self.setup.encoder_xy_to_w = EncoderXYToWLessClassNoVar(input_size, n_classes, w_size).to(DEVICE)
            elif model_config['xywsplit'] == 2:
                self.setup.encoder_xy_to_w = EncoderXYToWEvenLessClassNoVar(input_size, n_classes, w_size).to(DEVICE)
            elif model_config['xywsplit'] == 3:
                self.setup.encoder_xy_to_w = EncoderXYToWNoClassNoVar(input_size, n_classes, w_size).to(DEVICE)
            else:
                raise ValueError(f'Unknown xy w encoder split: {model_config["xywsplit"]}')
        else:
            self.setup.encoder_xy_to_w = EncoderXYToWNoVar(input_size, n_classes, w_size).to(DEVICE)

        if conv:
            self.setup.encoder_y_to_w = EncoderYToWNoVarConv(n_classes, w_size).to(DEVICE)
        else:
            self.setup.encoder_y_to_w = EncoderYToWNoVar(n_classes, w_size).to(DEVICE)

        if 'zwx' in model_config:
            if model_config['zwx'] == 1:
                self.setup.decoder_zw_to_x = DecoderZWToXNoVar(z_size, w_size, self.in_channels).to(DEVICE)
            if model_config['zwx'] == 2:
                if conv:
                    self.setup.decoder_zw_to_x = DecoderZWToXOtherNoVarConv(z_size, w_size, self.in_channels).to(DEVICE)
                else:
                    self.setup.decoder_zw_to_x = DecoderZWToXOtherNoVar(z_size, w_size, self.in_channels).to(DEVICE)
        else:
            self.setup.decoder_zw_to_x = DecoderZWToXNoVar(z_size, w_size, self.in_channels).to(DEVICE)

        if conv:
            self.setup.decoder_z_to_y = DecoderZToYNoVarConv(z_size, 2).to(DEVICE)
        else:
            self.setup.decoder_z_to_y = DecoderZToYNoVar(z_size, 2).to(DEVICE)

        self.setup.csvae = CSAE(self.setup, DEVICE)

    def _is_default(self, type):
        return type == 'default'

    def _get_loader(self, execution_type):
        dataset_config = self.config[DATASET]

        type = dataset_config['type']
        in_memory = dataset_config['in_memory']
        batch_size = self.config['batch_size']
        self.setup.batch_size = batch_size

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

    def _setup_optimizers(self):
        lr = float(self.config['lr']['initial'])

        main_params = [param for param in self.setup.encoder_x_to_z.parameters()] + \
            [param for param in self.setup.encoder_xy_to_w.parameters()] + \
            [param for param in self.setup.decoder_zw_to_x.parameters()] + \
            [param for param in self.setup.encoder_y_to_w.parameters()]
        delta_params = self.setup.decoder_z_to_y.parameters()

        self.setup.main_optimizer = Adam(main_params, lr)
        self.setup.delta_optimizer = Adam(delta_params, lr)

        self.setup.main_scheduler = MultiStepLR(
            self.setup.main_optimizer,
            milestones=[pow(3, i) for i in range(7)],
            gamma=pow(0.1, 1/7),
        )
        self.setup.delta_scheduler = MultiStepLR(
            self.setup.delta_optimizer,
            milestones=[pow(3, i) for i in range(7)],
            gamma=pow(0.1, 1/7),
        )

    def _setup_loss_parameters(self):
        opt_config = self.config['optimizer']
        self.setup.beta1 = opt_config['beta1']
        self.setup.beta2 = opt_config['beta2']
        self.setup.beta3 = opt_config['beta3']
        self.setup.beta4 = opt_config['beta4']
        self.setup.beta5 = opt_config['beta5']

        x_recon_loss = opt_config['x_recon_loss']
        if x_recon_loss == 'mse':
            self.setup.x_recon_loss_function = nn.MSELoss()
        elif x_recon_loss == 'ssim':
            self.setup.x_recon_loss_function = SSIMLossSimilarityEncouraged(1, DEVICE, 1) # channels = 1
        else:
            raise ValueError(f'Unknown loss function: {x_recon_loss}')

