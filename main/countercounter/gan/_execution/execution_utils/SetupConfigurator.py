import random

import torch
from piqa.ssim import SSIM
from torch import nn

from main.countercounter.classifier.dataset.DataLoaderFashionMNIST import DataFashionMNIST
from main.countercounter.classifier.dataset.DataLoaderFashionMasks import DataMasks
from main.countercounter.classifier.dataset.DataLoaderMNIST import DataMNIST
from main.countercounter.classifier.dataset.DataLoaderRSNA import DataRSNA
from main.countercounter.classifier.dataset.DataSet import ExecutionType
from main.countercounter.gan._execution.execution_utils.Setup import Setup
from main.countercounter.gan.autoencoder.Autoencoder import Autoencoder
from main.countercounter.gan.autoencoder.AutoencoderLoss import AutoencoderLoss
from main.countercounter.gan.cGAN.discriminator.d_patch_gan import DPatchGan
from main.countercounter.gan.cGAN.discriminator.d_patch_gan_small import DPatchGanSmall, DPatchGanBig, DPatchGanHuge, \
    DPatchGanHuger
from main.countercounter.gan.cGAN.discriminator.d_patch_gan_small_classification import DPatchGanSmallWithClass, \
    DPatchGanWithClass, DPatchGanBigWithClass, DPatchGanHugeWithClass, DPatchGanHugerWithClass
from main.countercounter.gan.cGAN.generator.custom_generator import G
from main.countercounter.gan.cGAN.generator.g_pix2pix import GPix2Pix
from main.countercounter.gan.cGAN.generator.g_pix2pix_no_unet_small import GPix2PixSmallNoUnet
from main.countercounter.gan.cGAN.generator.g_pix2pix_small import GPix2PixSmall
from main.countercounter.gan.cGAN.generator.g_pix2pix_small_labels import GPix2PixSmallLabels
from main.countercounter.gan.cGAN.loss.discriminator_loss_calculator import DLossCalculatorPatchGAN, \
    DualDLossCalculatorPatchGAN
from main.countercounter.gan.cGAN.loss.generator_loss_calculator import GLossCalculatorCounter, \
    DualGLossCalculatorCounter
from main.countercounter.gan.cGAN.loss.loss import DiscriminatorLoss, GeneratorLoss
from main.countercounter.gan.cGAN.loss.loss_serialization import LossPrinterPix2Pix, LossTensorboardLoggerPix2Pix
from main.countercounter.gan.classifier.Classifier import Classifier
from main.countercounter.gan.classifier.ClassifierLoss import MSELoss, InvertedMSELoss, ArgmaxLoss, InvertedArgmaxLoss, \
    BCELossWrapper, InvertedBCELossWrapper, InvertedAbsDistLoss, AbsDistLoss
from main.countercounter.gan.csae.CSAE import CSAE
from main.countercounter.gan.ssim.MSELossDissimilarityEncouraged import MSELossDissimilarityEncouraged
from main.countercounter.gan.ssim.MSELossSimilarityEncouraged import MSELossSimilarityEncouraged
from main.countercounter.gan.ssim.SSIMLossDissimilarityEncouraged import SSIMLossDissimilarityEncouraged
from main.countercounter.gan.ssim.SSIMLossSimilarityEncouraged import SSIMLossSimilarityEncouraged
from main.countercounter.gan.svm.SVM import SVM
from main.countercounter.gan.utils.AbstractTraining import DEVICE

DIR = 'dir'

CHECKPOINTS = 'checkpoints'
CLASSIFIER = 'classifier'
AUTOENCODER = 'autoencoder'
DATASET = 'dataset'
GENERATOR = 'generator'
DISCRIMINATOR = 'discriminator'

WEIGHT = 'weight_init'
MEAN = 'mean'
STD = 'std'


class SetupConfigurator:

    def __init__(self, config, root_dir, config_nr, eval=False, path_to_dataset=None, add_identifier=False, size_check=True, path_to_kdes=None, shuffle=True):
        self.root_dir = root_dir
        self.config_nr = config_nr
        self.config = config
        self.path_to_dataset = path_to_dataset
        self.path_to_kdes = path_to_kdes

        self.class_to_generate = None

        self.channels = 1

        self.eval = eval

        self.setup = Setup()

        self.add_identifier = add_identifier

        self.size_check = size_check

        self.shuffle = shuffle
        if not shuffle:  # for deterministic data loaders
            random.seed(42)
            torch.manual_seed(42)

    def configure(self) -> Setup:
        self.setup.minimal_logging = self.config['minimal_logging']

        self.class_to_generate = self.config['losses']['class_to_generate']
        if self.class_to_generate != 0 and self.class_to_generate != 1:
            self.class_to_generate = None

        self.setup.classifier = self._get_classifier()
        self.setup.autoencoder = self._get_autoencoder()
        self._setup_models()

        self.setup.generator_optimizer = self._get_optimizer(GENERATOR, self.setup.generator)
        self.setup.discriminator_optimizer = self._get_optimizer(DISCRIMINATOR, self.setup.discriminator)
        self.setup.discriminator_plausibility_optimizer = self._get_optimizer('plausibility', self.setup.plausibility_discriminator)

        self.setup.ssim = self._get_ssim()

        self.setup.train_loader = self._get_loader(ExecutionType.TRAIN)
        self.setup.val_loader = self._get_loader(ExecutionType.VAL)

        if self.eval:
            self.setup.test_loader = self._get_loader(ExecutionType.TEST)
            # self.setup.full_set_loader = self._get_full_loader()

        self._setup_lambdas()

        self._setup_csae()
        self.setup.svm = self._get_svm()

        self.setup.epochs = self.config['epochs']

        self.setup.checkpoints_dir = self._add_root_dir(self.config[DIR]['checkpoints'])
        self.setup.root_dir = self.root_dir
        self.setup.config_nr = self.config_nr
        self.setup.tensorboard_dir = self._add_root_dir(self.config[DIR]['tensorboard'])
        self.setup.model_dir = self._add_root_dir(self.config[DIR]['model'])
        self.setup.image_dir = self._add_root_dir(self.config[DIR]['image'])

        self.setup.image_sample_size = self.config['image_sample_size']

        self.setup.ssim_function = self._ssim_function()

        self._setup_losses()
        self._setup_loss_logger()

        return self.setup

    def _setup_lambdas(self):
        self.setup.lambda_classifier = self.config['lambdas'][CLASSIFIER]
        self.setup.lambda_ssim = self.config['lambdas']['ssim']

        if 'svm' in self.config['lambdas']:
            self.setup.lambda_svm = self.config['lambdas']['svm']
        else:
            self.setup.lambda_svm = 0

        if 'autoencoder' in self.config['lambdas']:
            self.setup.lambda_autoencoder = self.config['lambdas']['autoencoder']
        else:
            self.setup.lambda_autoencoder = 0
        if 'kde' in self.config['lambdas']:
            self.setup.lambda_kde = self.config['lambdas']['kde']
        else:
            self.setup.lambda_kde = 0

        if 'csae' in self.config['lambdas']:
            self.setup.lambda_csae = self.config['lambdas']['csae']
        else:
            self.setup.lambda_csae = 0

    def _setup_losses(self):
        wgan = False
        if 'wgan' in self.config['models'] and self.config['models']['wgan']:
            self.setup.weight_clipping = self.config['models']['clipping'] is not None
            self.setup.clipping = self.config['models']['clipping']

            wgan = True

        ssim_epoch_escalation = False
        escalation_by_epoch = 0

        loss_config = self.config['losses']
        if 'ssim_epoch_escalation' in loss_config and loss_config['ssim_epoch_escalation']:
            ssim_epoch_escalation = True
            escalation_by_epoch = loss_config['escalation_by_epoch']

        self.setup.generator_loss = GeneratorLoss(DEVICE)

        if self.setup.plausibility_discriminator is not None:
            self.setup.plausibility_generator_loss = GeneratorLoss(DEVICE)
            self.setup.generator_loss_calculator = DualGLossCalculatorCounter(
                DEVICE,
                self.setup.generator_loss,
                self.setup.plausibility_generator_loss,
                self.setup.classifier,
                self.setup.ssim,
                self.setup.autoencoder,
                self.setup.kde,
                self.setup.lambda_classifier,
                self.setup.lambda_ssim,
                self.setup.lambda_autoencoder,
                self.setup.lambda_kde,
                self.setup.one_sided_label_smoothing,
                self.setup.lambda_plausibility_discriminator,
                self.setup.one_sided_label_smoothing_plausibility,
                wgan,
                ssim_epoch_escalation,
                escalation_by_epoch,
                csae=self.setup.csae,
                lambda_csae=self.setup.lambda_csae,
                svm=self.setup.svm,
                lambda_svm=self.setup.lambda_svm,
            )
        else:
            self.setup.generator_loss_calculator = GLossCalculatorCounter(
                DEVICE,
                self.setup.generator_loss,
                self.setup.classifier,
                self.setup.ssim,
                self.setup.autoencoder,
                self.setup.kde,
                self.setup.lambda_classifier,
                self.setup.lambda_ssim,
                self.setup.lambda_autoencoder,
                self.setup.lambda_kde,
                self.setup.one_sided_label_smoothing,
                wgan,
                ssim_epoch_escalation,
                escalation_by_epoch,
                csae=self.setup.csae,
                lambda_csae=self.setup.lambda_csae,
                svm=self.setup.svm,
                lambda_svm=self.setup.lambda_svm,
            )

        dis_type = self.config['models'][DISCRIMINATOR]['type']
        if self._is_discriminator_patchgan(dis_type) or self._is_discriminator_patchgan_small(dis_type) or self._is_discriminator_patchgan_small_with_classification(dis_type)\
                or self._is_discriminator_patchgan_with_classification(dis_type) or self._is_discriminator_patchgan_big_with_classification(dis_type) or self._is_discriminator_patchgan_big(dis_type)\
                or self._is_discriminator_patchgan_huge(dis_type) or self._is_discriminator_patchgan_huge_with_classification(dis_type)\
                or self._is_discriminator_patchgan_huger(dis_type) or self._is_discriminator_patchgan_huger_with_classification(dis_type):
            self.setup.discriminator_loss = DiscriminatorLoss(DEVICE)
            self.setup.discriminator_loss_calculator = DLossCalculatorPatchGAN(
                DEVICE,
                self.setup.discriminator_loss,
                self.setup.one_sided_label_smoothing,
                wgan,
            )

            if self.setup.plausibility_discriminator is not None:
                self.setup.discriminator_loss_plausibility = DiscriminatorLoss(DEVICE)

            self.setup.discriminator_loss_calculator = DualDLossCalculatorPatchGAN(
                DEVICE,
                self.setup.discriminator_loss,
                self.setup.one_sided_label_smoothing,
                self.setup.discriminator_loss_plausibility,
                self.setup.one_sided_label_smoothing_plausibility,
                wgan,
            )

    def _setup_models(self):
        input_size = int(self.config[DIR][DATASET][-3:])

        model_config = self.config['models']

        gen_config = model_config[GENERATOR]
        dis_config = model_config[DISCRIMINATOR]

        self.setup.one_sided_label_smoothing = dis_config['one_sided_label_smoothing']
        if 'one_sided_label_smoothing_plausibility' in dis_config:
            self.setup.one_sided_label_smoothing_plausibility = dis_config['one_sided_label_smoothing_plausibility']

        n_classes = gen_config['n_classes']

        more_noise = gen_config['more_noise']

        if 'channels' in self.config[DATASET]:
            self.channels = self.config[DATASET]['channels']

        if self._is_generator_pix2pix():
            use_conv2dtranspose = gen_config['use_conv2dtranspose']
            upsample_mode = gen_config['upsample_mode']
            self.setup.generator = GPix2Pix(in_channels=self.channels, out_channels=self.channels,
                                            input_size=input_size, use_conv2dtranspose=use_conv2dtranspose,
                                            upsample_mode=upsample_mode)
        elif self._is_generator_pix2pix_small():
            use_conv2dtranspose = gen_config['use_conv2dtranspose']
            upsample_mode = gen_config['upsample_mode']
            if n_classes is None or n_classes == 'None':
                self.setup.generator = GPix2PixSmall(in_channels=self.channels, out_channels=self.channels,
                                                input_size=input_size, use_conv2dtranspose=use_conv2dtranspose,
                                                upsample_mode=upsample_mode, more_noise=more_noise)
            else:
                self.setup.generator = GPix2PixSmallLabels(in_channels=self.channels, out_channels=self.channels,
                                                     input_size=input_size, use_conv2dtranspose=use_conv2dtranspose,
                                                     upsample_mode=upsample_mode, n_classes=n_classes)
        elif self._is_generator_pix2pix_small_no_unet():
            use_conv2dtranspose = gen_config['use_conv2dtranspose']
            upsample_mode = gen_config['upsample_mode']
            self.setup.generator = GPix2PixSmallNoUnet(in_channels=self.channels, out_channels=self.channels,
                                                 input_size=input_size, use_conv2dtranspose=use_conv2dtranspose,
                                                 upsample_mode=upsample_mode, more_noise=more_noise)
        elif self._is_generator_custom():
            self.setup.generator = G(in_channels=self.channels, out_channels=self.channels,
                                                 input_size=input_size)
        else:
            raise ValueError(f'Unknown generator type')

        dis_type = self.config['models'][DISCRIMINATOR]['type']

        discriminator, set_labels = self._get_discriminator(input_size, dis_type)
        self.setup.discriminator = discriminator

        # second discriminator for plausibility
        dis_type = self.config['models'][DISCRIMINATOR]
        if 'dual' in dis_type and dis_type['dual']:
            self.setup.lambda_plausibility_discriminator = self.config['lambdas']['plausibility']
            self.setup.one_sided_label_smoothing_plausibility = dis_type['one_sided_label_smoothing_plausibility']
            type = dis_type['plausibility_type']
            discriminator, set_labels = self._get_discriminator(input_size, type)

            self.setup.plausibility_discriminator = discriminator

        if set_labels:
            self.setup.pass_labels_to_discriminator = True
            self.setup.discriminator_input_size = input_size

        self._initialize_weights(gen_config, dis_config)
        self.setup.generator = self.setup.generator.to(DEVICE)
        self.setup.discriminator = self.setup.discriminator.to(DEVICE)

        if self.setup.plausibility_discriminator is not None:
            self.setup.plausibility_discriminator = self.setup.plausibility_discriminator.to(DEVICE)

    def _get_discriminator(self, input_size, dis_type):
        if self._is_discriminator_patchgan(dis_type):
            return DPatchGan(in_channels=self.channels, input_size=input_size), False
        elif self._is_discriminator_patchgan_small(dis_type):
            return DPatchGanSmall(in_channels=self.channels, input_size=input_size), False
        elif self._is_discriminator_patchgan_small_with_classification(dis_type):
            return DPatchGanSmallWithClass(in_channels=self.channels + 1, input_size=input_size), True
        elif self._is_discriminator_patchgan_with_classification(dis_type):
            return DPatchGanWithClass(in_channels=self.channels + 1, input_size=input_size), True
        elif self._is_discriminator_patchgan_big_with_classification(dis_type):
            return DPatchGanBigWithClass(in_channels=self.channels + 1, input_size=input_size), True
        elif self._is_discriminator_patchgan_big(dis_type):
            return DPatchGanBig(in_channels=self.channels, input_size=input_size), False
        elif self._is_discriminator_patchgan_huge(dis_type):
            return DPatchGanHuge(in_channels=self.channels, input_size=input_size), False
        elif self._is_discriminator_patchgan_huge_with_classification(dis_type):
            return DPatchGanHugeWithClass(in_channels=self.channels + 1, input_size=input_size), True
        elif self._is_discriminator_patchgan_huger(dis_type):
            return DPatchGanHuger(in_channels=self.channels, input_size=input_size), False
        elif self._is_discriminator_patchgan_huger_with_classification(dis_type):
            return DPatchGanHugerWithClass(in_channels=self.channels + 1, input_size=input_size), True
        else:
            raise ValueError(f'Unknown discriminator type')

    def _get_optimizer(self, key, model):
        if key not in self.config['optimizer'] or model is None:
            return None

        opt_config = self.config['optimizer'][key]
        name = opt_config['name']
        lr = float(self.config['lr'][key])

        if name == 'Adam':
            betas = (opt_config['b1'], opt_config['b2'])
            return torch.optim.Adam(model.parameters(), lr, betas)

        return None

    def _get_classifier(self):
        classifier_config = self.config[CLASSIFIER]

        name = classifier_config['model']
        dir = classifier_config['dir']
        size = classifier_config['size']
        loss_class = self.config['losses']['classifier_loss']
        n_classes = self.config[DATASET]['n_classes']

        path = self._add_root_dir(dir) + '/' + f'{name}'

        self.setup.same_class = self.config['losses']['classifier_encouraged'] == 'similar'

        loss = self._get_classifier_loss(loss_class, self.setup.same_class, self.class_to_generate)

        in_channels = self.config[DATASET]['channels']

        return Classifier(path, DEVICE, size, loss, n_classes, in_channels)

    def _get_svm(self):
        if 'svm' not in self.config:
            return None

        svm_config = self.config['svm']

        name = svm_config['model']
        dir = svm_config['dir']

        path = self._add_root_dir(dir) + '/' + f'{name}'

        return SVM(path)

    def _get_ssim(self):
        if 'mse_ssim' in self.config['losses'] and self.config['losses']['mse_ssim']:
            ssim_loss_type = self.config['losses']['ssim_encouraged']

            if ssim_loss_type == 'similar':
                return MSELossSimilarityEncouraged(DEVICE)
            elif ssim_loss_type == 'dissimilar':
                return MSELossDissimilarityEncouraged(DEVICE)
            else:
                raise ValueError(f'Unknown mse loss type: {ssim_loss_type}')
        else:
            ssim_loss_type = self.config['losses']['ssim_encouraged']

            if ssim_loss_type == 'similar':
                return SSIMLossSimilarityEncouraged(1, DEVICE, self.channels)
            elif ssim_loss_type == 'dissimilar':
                return SSIMLossDissimilarityEncouraged(1, DEVICE, self.channels)
            else:
                raise ValueError(f'Unknown ssim loss type: {ssim_loss_type}')

    def _get_loader(self, execution_type):
        batch_size, dir, fake_second_class, in_memory, n_classes, type = self._get_dataset_info()
        return self._get_loader_class(batch_size, dir, execution_type, fake_second_class, in_memory, n_classes, type)

    def _get_dataset_info(self):
        dataset_config = self.config[DATASET]
        type = dataset_config['type']
        in_memory = dataset_config['in_memory']
        batch_size = self.config['batch_size']
        n_classes = dataset_config['n_classes']
        fake_second_class = dataset_config['fake_second_class']
        if fake_second_class:
            n_classes = 1
        if self.path_to_dataset is not None:
            dir = self.path_to_dataset
        else:
            dir = self._add_root_dir(self.config[DIR]['dataset'])
        return batch_size, dir, fake_second_class, in_memory, n_classes, type

    def _get_loader_class(self, batch_size, dir, execution_type, fake_second_class, in_memory, n_classes, type, full_set=False):
        if type == 'MNIST':
            return DataMNIST().get_loader(
                execution_type=execution_type,
                batch_size=batch_size,
                num_workers=2,
                in_memory=in_memory,
                root_dir=dir,
                fake_class=fake_second_class,  # only use in GAN training
                full_set=full_set,
                add_identifier=self.add_identifier,
                size_check=self.size_check,
                shuffle=self.shuffle,
            )
        if type == 'FashionMNIST':
            return DataFashionMNIST().get_loader(
                execution_type=execution_type,
                batch_size=batch_size,
                num_workers=2,
                in_memory=in_memory,
                root_dir=dir,
                full_set=full_set,
                add_identifier=self.add_identifier,
                size_check=self.size_check,
                shuffle=self.shuffle,
            )
        if type == 'MASKS':
            return DataMasks().get_loader(
                execution_type=execution_type,
                batch_size=batch_size,
                num_workers=2,
                in_memory=in_memory,
                root_dir=dir,
                full_set=full_set,
                size_check=self.size_check,
                add_identifier=self.add_identifier,
                shuffle=self.shuffle,
            )
        elif type == 'RSNA':
            return DataRSNA().get_loader(
                execution_type=execution_type,
                batch_size=batch_size,
                num_workers=2,
                in_memory=in_memory,
                root_dir=dir,
                full_set=full_set,
                add_identifier=self.add_identifier,
                size_check=self.size_check,
                shuffle=self.shuffle,
            )
        else:
            raise ValueError()

    def _is_grayscale(self, type):
        return 'Gray' in type

    def _add_root_dir(self, path):
        return self.root_dir + '/' + path

    def _initialize_weights(self, gen_config, dis_config):
        if WEIGHT in gen_config:
            self.setup.generator.apply(self.weights_init_g)

        if WEIGHT in dis_config:
            self.setup.discriminator.apply(self.weights_init_d)

            if self.setup.plausibility_discriminator is not None:
                self.setup.plausibility_discriminator.apply(self.weights_init_d)

    def weights_init_g(self, m):
        model_config = self.config['models']
        gen_config = model_config[GENERATOR]

        g_mean = gen_config[WEIGHT][MEAN]
        g_std = gen_config[WEIGHT][STD]

        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, g_mean, g_std)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, g_std)
            nn.init.constant_(m.bias.data, 0)

    def weights_init_d(self, m):
        model_config = self.config['models']
        dis_config = model_config[DISCRIMINATOR]
        d_mean = dis_config[WEIGHT][MEAN]
        d_std = dis_config[WEIGHT][STD]

        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, d_mean, d_std)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, d_std)
            nn.init.constant_(m.bias.data, 0)

    def _is_generator_pix2pix(self):
        gen_type = self._get_generator_type()
        return gen_type == 'Pix2Pix'

    def _is_generator_pix2pix_small(self):
        gen_type = self._get_generator_type()
        return gen_type == 'Pix2PixSmall'

    def _get_generator_type(self):
        gen_type = self.config['models'][GENERATOR]['type']
        return gen_type

    def _get_discriminator_type(self):
        dis_type = self.config['models'][DISCRIMINATOR]['type']
        return dis_type

    def _is_discriminator_patchgan(self, dis_type):
        return dis_type == 'PatchGAN'

    def _is_discriminator_patchgan_small(self, dis_type):
        return dis_type == 'PatchGANSmall'

    def _setup_loss_logger(self):
        self.setup.loss_printer = LossPrinterPix2Pix(dual=self.setup.plausibility_discriminator is not None)

        if not self.setup.minimal_logging:
            self.setup.loss_tensorboard_logger = LossTensorboardLoggerPix2Pix(dual=self.setup.plausibility_discriminator is not None)

    def _get_classifier_loss(self, loss_class, similarity_encouraged, class_to_generate=None):
        if loss_class == 'MSE':
            return self._loss_class(similarity_encouraged, MSELoss, InvertedMSELoss, class_to_generate)
        elif loss_class == 'Argmax':
            return self._loss_class(similarity_encouraged, ArgmaxLoss, InvertedArgmaxLoss, class_to_generate)
        elif loss_class == 'BCE':
            return self._loss_class(similarity_encouraged, BCELossWrapper, InvertedBCELossWrapper, class_to_generate)
        elif loss_class == 'AbsDist':
            return self._loss_class(similarity_encouraged, AbsDistLoss, InvertedAbsDistLoss, class_to_generate)
        else:
            raise ValueError(f'Unknown loss class {loss_class}')

    def _loss_class(self, similarity_encouraged, similar_loss, dissimilar_loss, class_to_generate):
        if similarity_encouraged:
            return similar_loss(class_to_generate)
        else:
            return dissimilar_loss(class_to_generate)

    def _is_generator_pix2pix_small_no_unet(self):
        gen_type = self.config['models'][GENERATOR]['type']
        return gen_type == 'Pix2PixSmallNoUnet'

    def _is_generator_custom(self):
        gen_type = self.config['models'][GENERATOR]['type']
        return gen_type == 'Custom1'

    def _is_discriminator_patchgan_small_with_classification(self, dis_type):
        return dis_type == 'PatchGANSmallClass'

    def _is_discriminator_patchgan_with_classification(self, dis_type):
        return dis_type == 'PatchGANClass'

    def _is_discriminator_patchgan_big_with_classification(self, dis_type):
        return dis_type == 'PatchGANBigClass'

    def _is_discriminator_patchgan_big(self, dis_type):
        return dis_type == 'PatchGANBig'

    def _is_discriminator_patchgan_huge(self, dis_type):
        return dis_type == 'PatchGANHuge'

    def _is_discriminator_patchgan_huge_with_classification(self, dis_type):
        return dis_type == 'PatchGANHugeClass'

    def _is_discriminator_patchgan_huger(self, dis_type):
        return dis_type == 'PatchGANHuger'

    def _is_discriminator_patchgan_huger_with_classification(self, dis_type):
        return dis_type == 'PatchGANHugerClass'

    def _get_full_loader(self):
        batch_size, dir, fake_second_class, in_memory, n_classes, type = self._get_dataset_info()
        return self._get_loader_class(batch_size, dir, None, fake_second_class, in_memory, n_classes, type, full_set=True)

    def configure_minimal(self):
        self.setup.classifier = self._get_classifier()
        return self.setup

    def _ssim_function(self):
        n_channels = 1
        if 'channels' in self.config[DATASET]:
            self.channels = self.config[DATASET]['channels']
            n_channels = self.channels

        return SSIM(value_range=1, n_channels=n_channels, non_negative=True)

    def _get_autoencoder(self):
        if not AUTOENCODER in self.config:
            return None

        classifier_config = self.config[AUTOENCODER]

        name = classifier_config['model']
        dir = classifier_config['dir']
        size = classifier_config['size']

        path = self._add_root_dir(dir) + '/' + f'{name}'

        loss = AutoencoderLoss()

        in_channels = self.config[DATASET]['channels']

        return Autoencoder(path, DEVICE, size, loss, in_channels)

    def _setup_csae(self):
        if 'csae' not in self.config or not self.config['csae']['use']:
            self.setup.csae = CSAE(None, None, None, None)
            return

        csae_config = self.config['csae']

        name = csae_config['model']
        dir = csae_config['dir']
        n_classes = self.config[DATASET]['n_classes']

        path = self._add_root_dir(dir) + '/' + f'{name}'

        in_channels = self.config[DATASET]['channels']

        self.setup.csae = CSAE(path, DEVICE, n_classes, in_channels)
