import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch

from main.countercounter.classifier.training.ConfigLoader import ConfigLoader
from main.countercounter.classifier.training.ModelLoader import ModelLoader
from main.countercounter.classifier.training.SetupConfigurator import SetupConfigurator

config_nr = '004'
root_dir = 'TODO'
config_dir = 'EmClass/configs'

checkpoint = 'epoch_15'
save_file = 'resnet18_trained.pt'


if __name__ == '__main__':
    config = ConfigLoader(root_dir, config_dir).load(config_nr)
    setup = SetupConfigurator(config, root_dir, config_nr, eval=True).configure()

    ModelLoader(setup).load(checkpoint)

    model = setup.model.to(torch.device('cpu'))

    torch.save(model, save_file)