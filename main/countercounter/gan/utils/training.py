import numpy as np
import torch

from main.countercounter.classifier.dataset.DataLoader import denormalize
from main.countercounter.gan._execution.execution_utils.LossItems import GDCSLossItems
from main.countercounter.gan._execution.execution_utils.Setup import Setup
from main.countercounter.gan.utils.AbstractTraining import AbstractTraining
from main.countercounter.gan.utils.img_utils import save_images


class Training(AbstractTraining):

    def __init__(self, setup: Setup):
        super().__init__(setup.tensorboard_dir, setup.model_dir, setup.checkpoints_dir, setup.minimal_logging)
        self.generator = setup.generator
        self.discriminator = setup.discriminator
        self.plausibility_discriminator = setup.plausibility_discriminator

        self.generator_optimizer = setup.generator_optimizer
        self.discriminator_optimizer = setup.discriminator_optimizer
        self.discriminator_plausibility_optimizer = setup.discriminator_plausibility_optimizer

        self.generator_loss_calculator = setup.generator_loss_calculator
        self.discriminator_loss_calculator = setup.discriminator_loss_calculator

        self.train_loader = setup.train_loader
        self.val_loader = setup.val_loader

        self.epochs = setup.epochs

        self.image_dir = setup.image_dir
        self.image_sample_size = setup.image_sample_size

        self.loss_printer = setup.loss_printer
        self.loss_tensorboard_logger = setup.loss_tensorboard_logger

        self.classifier = setup.classifier
        self.pass_labels_to_discriminator = setup.pass_labels_to_discriminator
        self.discriminator_input_size = setup.discriminator_input_size
        self.same_class = setup.same_class

        self.minimal_logging = setup.minimal_logging

        self.weight_clipping = setup.weight_clipping
        self.clipping = setup.clipping

        self.use_svm = setup.svm is not None

    def train(self):
        self._initialize()
        self._store_sample_images('__')

        self._eval_losses_before()
        for epoch in range(self.epochs):
            epoch_losses_generator = []
            epoch_losses_discriminator = []
            epoch_losses_plausibility = []

            for real_data, labels in self.train_loader:
                real_data = real_data.to(self.device)
                labels = labels.to(self.device)

                loss_discriminator, loss_plausibility = self.train_discriminator(real_data, labels)
                loss_items_generator = self.train_generator(real_data, labels, epoch)

                epoch_losses_discriminator.append(loss_discriminator)
                epoch_losses_generator.append(loss_items_generator)
                epoch_losses_plausibility.append(loss_plausibility)

            self._save_checkpoint(epoch)

            self._store_sample_images(str(epoch))
            self._log(epoch, epoch_losses_discriminator, epoch_losses_generator, epoch_losses_plausibility)

        torch.save(self.discriminator, f'{self.model_dir}/discriminator_{self.start_time:%Y_%m_%d_%H_%M}.pt')
        if self.plausibility_discriminator is not None:
            torch.save(self.plausibility_discriminator, f'{self.model_dir}/discriminator_plausibility_{self.start_time:%Y_%m_%d_%H_%M}.pt')
        torch.save(self.generator, f'{self.model_dir}/generator{self.start_time:%Y_%m_%d_%H_%M}.pt')

        self.writer.close()

    def _log(self, epoch, losses_discriminator, losses_generator, losses_plausibility):
        self.loss_printer.initialize(losses_discriminator, losses_generator, losses_plausibility)

        if not self.minimal_logging:
            self.loss_tensorboard_logger.initialize(losses_discriminator, losses_generator, losses_plausibility)

        print(f'[{epoch:03}/{self.epochs:03}]: {self.loss_printer.print()}')
        self._log_tensorboard(epoch)

    def _initialize(self):
        self.generator.train()
        self.discriminator.train()

    def train_discriminator(self, real_data, labels):
        self.discriminator_optimizer.zero_grad()

        for p in self.discriminator.parameters():
            p.requires_grad = True

        if self.discriminator_plausibility_optimizer is not None:
            self.discriminator_plausibility_optimizer.zero_grad()

            for p in self.plausibility_discriminator.parameters():
                p.requires_grad = True

        if self.pass_labels_to_discriminator:
            original_label = self._get_label(real_data)

            if self.plausibility_discriminator is None:
                real_discriminator_prediction = self.discriminator(real_data, original_label)
            else:
                real_discriminator_prediction = self.discriminator(real_data)
        else:
            real_discriminator_prediction = self.discriminator(real_data)

        if self.plausibility_discriminator is not None:
            real_plausibility_prediction = self.plausibility_discriminator(real_data, original_label)
        else:
            real_plausibility_prediction = None

        fake_data = self.generator(real_data, labels).detach()

        if self.pass_labels_to_discriminator:
            if self.same_class:
                gen_label = self._get_label(real_data)
            else:
                gen_label = self._get_flipped_label(real_data)

            if self.plausibility_discriminator is None:
                fake_discriminator_prediction = self.discriminator(fake_data, gen_label)
            else:
                fake_discriminator_prediction = self.discriminator(fake_data)

            if self.plausibility_discriminator is not None:
                fake_plausibility_prediction = self.plausibility_discriminator(fake_data, gen_label)
            else:
                fake_plausibility_prediction = None
        else:
            fake_discriminator_prediction = self.discriminator(fake_data)
            fake_plausibility_prediction = None

        if self.plausibility_discriminator is not None:
            self.discriminator_loss_calculator.calculate(
                real_discriminator_prediction,
                fake_discriminator_prediction,
                real_plausibility_prediction,
                fake_plausibility_prediction,
            )
        else:
            self.discriminator_loss_calculator.calculate(real_discriminator_prediction, fake_discriminator_prediction, None, None)

        loss = self.discriminator_loss_calculator.loss_tensor
        loss.backward()
        self.discriminator_optimizer.step()

        if self.plausibility_discriminator is not None:
            plausibility_loss = self.discriminator_loss_calculator.plausibility_loss_tensor
            plausibility_loss.backward()
            self.discriminator_plausibility_optimizer.step()

        self.discriminator_loss_calculator.reset_loss_tensor()

        if self.weight_clipping:
            for p in self.discriminator.parameters():
                p.data.clamp_(-self.clipping, self.clipping)

            if self.plausibility_discriminator is not None:
                for p in self.plausibility_discriminator.parameters():
                    p.data.clamp_(-self.clipping, self.clipping)

        if self.plausibility_discriminator is not None:
            return self.discriminator_loss_calculator.loss_stats, self.discriminator_loss_calculator.plausibility_loss_stats

        return self.discriminator_loss_calculator.loss_stats, 0

    def train_generator(self, real_data, labels, epoch) -> GDCSLossItems:
        for p in self.discriminator.parameters():
            p.requires_grad = False

        if self.plausibility_discriminator is not None:
            for p in self.plausibility_discriminator.parameters():
                p.requires_grad = False

        self.generator_optimizer.zero_grad()

        generated_data = self.generator(real_data, labels)

        activations_real, activations_generated = None, None
        if self.use_svm:
            activations_real, activations_generated = self.classifier.get_activations(real_data, generated_data)

        if self.same_class:
            gen_label = self._get_label(real_data)
        else:
            gen_label = self._get_flipped_label(real_data)

        if self.pass_labels_to_discriminator:
            if self.plausibility_discriminator is not None:
                discriminator_judgement = self.discriminator(generated_data)
                plausibility_judgement = self.plausibility_discriminator(generated_data, gen_label)
            else:
                discriminator_judgement = self.discriminator(generated_data, gen_label)
        else:
            discriminator_judgement = self.discriminator(generated_data)

        if self.plausibility_discriminator is not None:
            self.generator_loss_calculator.calculate(real_data, generated_data, discriminator_judgement, gen_label, plausibility_judgement, epoch, activations_real, activations_generated)
        else:
            self.generator_loss_calculator.calculate(real_data, generated_data, discriminator_judgement, gen_label, epoch, activations_real, activations_generated)

        loss = self.generator_loss_calculator.loss_tensor
        loss.backward()
        self.generator_optimizer.step()

        self.generator_loss_calculator.reset_loss_tensor()

        return self.generator_loss_calculator.loss_stats

    def _log_tensorboard(self, epoch):
        if not self.minimal_logging:
            for (variable_name, variable) in self.loss_tensorboard_logger.loggable_properties:
                self.writer.add_scalar(variable_name, variable, epoch)

            for tag, param in self.generator.named_parameters():
                self.writer.add_histogram('generator_' + tag, param, epoch)
                self.writer.add_histogram('generator_' + tag + '_grad', param.grad.data.cpu().detach().numpy(), epoch)

            for tag, param in self.discriminator.named_parameters():
                self.writer.add_histogram('discriminator_' + tag, param, epoch)
                self.writer.add_histogram('discriminator_' + tag + '_grad', param.grad.data.cpu().detach().numpy(), epoch)

    def _save_checkpoint(self, epoch):
        if not self.minimal_logging:
            torch.save({
                'epoch': epoch,
                'generator_model_state_dict': self.generator.state_dict(),
                'generator_optimizer_state_dict': self.generator_optimizer.state_dict(),
                'discriminator_model_state_dict': self.discriminator.state_dict(),
                'discriminator_optimizer_state_dict': self.discriminator_optimizer.state_dict(),
            }, f'{self.checkpoints_dir}/epoch_{epoch}')

    def _store_sample_images(self, epoch):
        images = 0

        originals = []
        generated_imgs = []
        generated_imgs_rescaled = []

        for idx, (real_data, labels) in enumerate(self.train_loader):
            if images >= self.image_sample_size:
                break

            batch_size = real_data.size(0)
            images += batch_size

            data = real_data.to(self.device)
            labels = labels.to(self.device)

            # self.generator.eval() according to Isola
            generated = self.generator(data, labels)

            # reverse normalization during import
            data = denormalize(data)
            generated_rescaled = denormalize(generated)
            generated_imgs_rescaled.append(generated_rescaled.cpu().detach())

            originals.append(data.cpu().detach())
            generated_imgs.append(generated.cpu().detach())

        save_images(epoch, originals, generated_imgs, generated_imgs_rescaled, self.image_dir, self.image_sample_size)

    def _get_flipped_label(self, original_image):
        return 1 - self._get_label(original_image)

    def _get_label(self, original_image):
        return self.classifier.get_class(original_image)

    def _eval_losses_before(self):
        for p in self.discriminator.parameters():
            p.requires_grad = False
        for p in self.generator.parameters():
            p.requires_grad = False

        svm_losses = []
        csae_losses = []

        for real_data, labels in self.train_loader:
            real_data = real_data.to(self.device)
            labels = labels.to(self.device)

            generated_data = self.generator(real_data, labels)

            activations_real, activations_generated = None, None
            if self.use_svm:
                activations_real, activations_generated = self.classifier.get_activations(real_data, generated_data)

            if self.same_class:
                gen_label = self._get_label(real_data)
            else:
                gen_label = self._get_flipped_label(real_data)

            if self.pass_labels_to_discriminator:
                discriminator_judgement = self.discriminator(generated_data, gen_label)
            else:
                discriminator_judgement = self.discriminator(generated_data)

            self.generator_loss_calculator.calculate(real_data, generated_data, discriminator_judgement, gen_label,
                                                     0, activations_real, activations_generated)

            self.generator_loss_calculator.reset_loss_tensor()

            svm, csae = self.generator_loss_calculator.loss_stats.svm, self.generator_loss_calculator.loss_stats.csae
            svm_losses.append(svm)
            csae_losses.append(csae)

        print(f'Before training: svm {np.mean(svm_losses)} -- csae {np.mean(csae_losses)}')

        for p in self.discriminator.parameters():
            p.requires_grad = True
        for p in self.generator.parameters():
            p.requires_grad = True