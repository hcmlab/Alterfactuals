import numpy as np
from torchvision.utils import save_image

from main.countercounter.classifier.dataset.DataLoaderMNIST import denormalize
from main.countercounter.gan.utils.AbstractTraining import DEVICE


class Evaluator:

    def __init__(self, setup, config_nr, epoch):
        self.val_loader = setup.val_loader
        self.test_loader = setup.test_loader

        self.model = setup.model
        self.model.eval()

        self.config_nr = config_nr
        self.epoch = epoch

        self.loss = setup.criterion

        self.print_counter = 0

    def evaluate(self):
        self._eval(self.test_loader, self.model, 'test', 'original')

    def _eval(self, data_loader, model, name, model_type):
        losses = []

        for data, _ in data_loader:
            data = data.to(DEVICE)
            preds = model(data)

            self.print(data, preds)

            loss = self.loss(preds, data)
            losses.append(loss.item())

        print(f'{name} Loss: {np.mean(losses):.6f}')

    def print(self, images, reconstructed):
        if self.print_counter > 10:
            return

        for idx in range(images.size(0)):
            img = images[idx]
            rec = reconstructed[idx]

            img_denorm = denormalize(img)
            rec_denorm = denormalize(rec)

            path_img = f'img_{self.print_counter}_orig'
            path_rec = f'img_{self.print_counter}_rec'

            save_image(img_denorm, path_img, format='png')
            save_image(rec_denorm, path_rec, format='png')

            self.print_counter += 1