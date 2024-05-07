import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from main.countercounter.classifier.evaluation.SVMEvaluationPipeline import SVMEvaluationPipeline

config_nr = '027'
root_dir = 'TODO'
config_dir = 'EmClass/configs'


if __name__ == '__main__':
    epoch = 2
    print(f'Epoch: {epoch}')
    SVMEvaluationPipeline().run(
        config_nr,
        f'epoch_{epoch}',
        path_to_output_folder='TODO',
        path_to_activation_train_csv='TODO',
        path_to_activation_val_csv='TODO',
    )
    print('-----------------')