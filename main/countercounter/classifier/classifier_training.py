import os
from os.path import sep

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from main.countercounter.classifier.training.TrainingsPipeline import TrainingsPipeline

config_nr = '030'
root_dir = 'TODO'
config_dir = f'EmClass{sep}configs'


if __name__ == '__main__':
    TrainingsPipeline().run(config_nr, root_dir, config_dir, path_to_dataset='TODO')