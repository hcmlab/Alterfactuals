import torch

from main.countercounter.classifier.training.Setup import Setup

import numpy as np

from main.countercounter.gan.utils.AbstractTraining import AbstractTraining


class Training(AbstractTraining):

    def __init__(self, setup: Setup):
        super().__init__(setup.tensorboard_dir, setup.model_dir, setup.checkpoints_dir)

        self.model = setup.model
        self.criterion = setup.criterion
        self.optimizer = setup.optimizer

        self.epochs = setup.epochs

        self.train_loader = setup.train_loader
        self.val_loader = setup.val_loader

    def train(self):
        self.model.train()

        val_loss = self._eval()
        print(f'Val Loss before training {val_loss}')

        for epoch in range(self.epochs):
            epoch_losses = []

            count = 0
            for data, _ in self.train_loader:
                if count >= 20:
                    break

                count += data.size(0)

                data = data.to(self.device)
                self.optimizer.zero_grad()

                preds = self.model(data)

                loss = self.criterion(preds, data)
                loss.backward()
                self.optimizer.step()

                epoch_losses.append(loss.item())

            epoch_loss = np.mean(epoch_losses)

            self.save_checkpoint(epoch)
            val_loss = self._eval()
            self._log(epoch, epoch_loss, val_loss)

    def _log(self, epoch, epoch_loss, val_loss):
        print(f'[{epoch:03}/{self.epochs:03}]: Train loss {epoch_loss:.6f} Val Loss {val_loss:.6f}')

        self.writer.add_scalar('loss', epoch_loss, epoch)
        self.writer.add_scalar('val acc', val_loss, epoch)

        for tag, param in self.model.named_parameters():
            self.writer.add_histogram(tag, param, epoch)

    def _eval(self):
        self.model.eval()

        loss = self.criterion

        eval_loss = []
        for data, _ in self.val_loader:
            data = data.to(self.device)

            preds = self.model(data)
            losses = loss(preds, data)

            eval_loss.append(losses.item())

        self.model.train()

        return np.mean(eval_loss)

    def save_checkpoint(self, epoch):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, f'{self.checkpoints_dir}/epoch_{epoch}')