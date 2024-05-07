import torch

import numpy as np
import torch.nn as nn

from main.countercounter.classifier.dataset.DataLoader import denormalize
from main.countercounter.csvae.networks.CSVAE import CSVAE
from main.countercounter.gan.classifier.Classifier import Classifier
from main.countercounter.gan.utils.AbstractTraining import AbstractTraining
from main.countercounter.csvae.training.Setup import Setup
from main.countercounter.gan.utils.img_utils import save_images


# generally inspired by the following repos
# https://github.com/kareenaaahuang/am207_final_project.git
# https://github.com/alexlyzhov/latent-subspaces.git


class Training(AbstractTraining):

    def __init__(self, setup: Setup):
        super().__init__(setup.tensorboard_dir, setup.model_dir, setup.checkpoints_dir)

        self.epochs = setup.epochs

        self.train_loader = setup.train_loader
        self.val_loader = setup.val_loader

        self.csvae: CSVAE = setup.csvae
        self.classifier: Classifier = setup.classifier

        self.main_optimizer = setup.main_optimizer
        self.delta_optimizer = setup.delta_optimizer

        self.main_scheduler = setup.main_scheduler
        self.delta_scheduler = setup.delta_scheduler

        self.image_sample_size = setup.image_sample_size
        self.image_dir = setup.image_dir

    def train(self):
        w_pred_acc = self._eval('__')
        print(f'Val pred loss before: {w_pred_acc:0.5f}')

        for epoch in range(self.epochs):
            self.csvae.train()

            main_losses = []
            delta_losses = []
            recon_losses = []
            m2_losses = []

            for data, _ in self.train_loader:
                delta_loss_item, main_loss_item, recon_loss_item, m2_item = self._train_step(data)

                main_losses.append(main_loss_item)
                delta_losses.append(delta_loss_item)
                recon_losses.append(recon_loss_item)
                m2_losses.append(m2_item)

            w_pred_acc = self._eval(epoch)

            self.save_checkpoint(epoch)
            print(f'[{epoch:03}/{self.epochs:03}] main loss: {np.mean(main_losses):0.5f} / x recon: {np.mean(recon_losses)} / m2: {np.mean(m2_losses)}, delta loss: {np.mean(delta_losses):0.5f} -- Pred Acc on val-y: {w_pred_acc:0.5f}')

            self.main_scheduler.step()
            self.delta_scheduler.step()

    def _train_step(self, data):
        x = data.to(self.device)
        y, y_scores = self.classifier.get_class_and_logits(x)

        main_loss, delta_loss, recon_loss, m2 = self.csvae.get_losses(x, y, y_scores)

        self.main_optimizer.zero_grad()
        self.delta_optimizer.zero_grad()

        main_loss.backward(retain_graph=True)
        delta_loss.backward()

        self.main_optimizer.step()
        self.delta_optimizer.step()

        main_loss_item = main_loss.detach().cpu().item()
        delta_loss_item = delta_loss.detach().cpu().item()
        recon_loss_item = recon_loss.detach().cpu().item()
        m2_item = m2.detach().cpu().item()

        del main_loss
        del delta_loss
        del recon_loss
        del m2
        return delta_loss_item, main_loss_item, recon_loss_item, m2_item

    def _eval(self, epoch):
        self.csvae.eval()

        originals = []
        generated_imgs = []
        generated_imgs_rescaled = []

        eval_accs = []

        for data, _ in self.val_loader:
            x = data.to(self.device)
            y, y_scores = self.classifier.get_class_and_logits(x)

            x_pred, y_pred = self.csvae.get_x_pred_y_pred(x, y, y_scores)

            if len(generated_imgs) < self.image_sample_size:
                data = denormalize(x)
                generated_rescaled = denormalize(x_pred)
                generated_imgs_rescaled.append(generated_rescaled.cpu().detach()[:self.image_sample_size])

                originals.append(data.cpu().detach()[:self.image_sample_size])
                generated_imgs.append(x_pred.cpu().detach()[:self.image_sample_size])

            _, classes = torch.max(y_pred, 1)
            equals = (classes == y).cpu().type(torch.FloatTensor)
            eval_accs.append(torch.mean(equals).item())

        self.csvae.train()
        save_images(epoch, originals, generated_imgs, generated_imgs_rescaled, self.image_dir, self.image_sample_size)

        return np.mean(eval_accs)

    def save_checkpoint(self, epoch):
        d = {
            'epoch': epoch,
            'main_optimizer_state_dict': self.main_optimizer.state_dict(),
            'delta_optimizer_state_dict': self.delta_optimizer.state_dict(),
            'main_lr_scheduler_state_dict': self.main_scheduler.state_dict(),
            'delta_lr_scheduler_state_dict': self.delta_scheduler.state_dict(),
        }

        for name, state_dict in self.csvae.model_checkpoint_info():
            d[f'{name}_csvae_state_dict'] = state_dict

        torch.save(d, f'{self.checkpoints_dir}/epoch_{epoch}')

