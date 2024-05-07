import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from main.countercounter.classifier.evaluation.EvaluationPipeline import EvaluationPipeline

config_nr = '030'
root_dir = 'TODO'
config_dir = 'EmClass/configs'


if __name__ == '__main__':
    for epoch in range(2, 3):
        print(f'Epoch: {epoch}')
        EvaluationPipeline().run(
            config_nr,
            root_dir,
            config_dir,
            f'epoch_{epoch}',
            path_to_dataset='TODO')
        print('-----------------')