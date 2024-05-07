import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from main.countercounter.csvae.evaluation.EvaluationPipeline import EvaluationPipeline

config_nr = '038'
root_dir = 'TODO'
config_dir = 'EmClass/CSVAEConfigs'

if __name__ == '__main__':
    for epoch in range(32, 33):
        EvaluationPipeline().run(
            config_nr,
            root_dir,
            config_dir,
            f'epoch_{epoch}',
            path_to_dataset='TODO',
        )
        print('-----------------')