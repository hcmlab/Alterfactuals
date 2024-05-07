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

        val_acc = self._eval()
        print(f'Val Acc before training {val_acc}')

        for epoch in range(self.epochs):
            epoch_losses = []
            epoch_accs = []

            for data, labels in self.train_loader:
                data = data.to(self.device)
                labels = labels.to(self.device).float()

                if len(labels.size()) == 1:
                    labels_reshaped = labels.reshape(labels.size(0), 1)
                else:
                    labels_reshaped = labels

                self.optimizer.zero_grad()

                preds = self.model(data)

                if preds.size(1) > 1:
                    _, classes = torch.max(preds, 1)
                    _, labels_reshaped = torch.max(labels, 1)
                else:
                    classes = (preds >= 0.5).float()
                    classes = classes.reshape(classes.size(0))

                loss = self.criterion(preds, labels_reshaped)
                loss.backward()

                self.optimizer.step()

                epoch_losses.append(loss.item())

                equals = (classes == labels_reshaped).cpu().type(torch.FloatTensor)
                epoch_accs.append(torch.mean(equals).item())

            epoch_loss = np.mean(epoch_losses)
            train_acc = np.mean(epoch_accs)

            self.save_checkpoint(epoch)
            val_acc = self._eval()
            self._log(epoch, epoch_loss, train_acc, val_acc)

    def _log(self, epoch, epoch_loss, train_acc, val_acc):
        print(f'[{epoch:03}/{self.epochs:03}]: Train loss {epoch_loss:.6f} Train Acc {train_acc:.6f} Val Acc {val_acc:.6f}')

        self.writer.add_scalar('loss', epoch_loss, epoch)
        self.writer.add_scalar('train acc', train_acc, epoch)
        self.writer.add_scalar('val acc', val_acc, epoch)

        for tag, param in self.model.named_parameters():
            self.writer.add_histogram(tag, param, epoch)

    def _eval(self):
        self.model.eval()

        eval_accs = []
        for data, labels in self.val_loader:
            data = data.to(self.device)
            labels = labels.to(self.device)

            if len(labels.size()) == 1:
                labels = labels.reshape(labels.size(0))

            preds = self.model(data)

            if preds.size(1) > 1:
                _, classes = torch.max(preds, 1)
                _, labels = torch.max(labels, 1)
            else:
                classes = (preds >= 0.5).float()
                classes = classes.reshape(classes.size(0))

            equals = (classes == labels).cpu().type(torch.FloatTensor)
            eval_accs.append(torch.mean(equals).item())

        self.model.train()

        return np.mean(eval_accs)

    def save_checkpoint(self, epoch):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, f'{self.checkpoints_dir}/epoch_{epoch}')