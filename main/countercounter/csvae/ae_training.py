import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from main.countercounter.csvae.training.TrainingsPipeline import TrainingsPipeline

config_nr = '000'
root_dir = 'TODO'
config_dir = 'configs'

if __name__ == '__main__':
    TrainingsPipeline().run(config_nr, root_dir, config_dir, path_to_dataset='TODO')