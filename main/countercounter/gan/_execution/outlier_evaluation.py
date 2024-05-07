import os
from os.path import sep

from main.countercounter.gan._execution.evaluation.outlier.OutlierEvaluationPipeline import OutlierEvaluationPipeline

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


root_dir = 'TODO'
config_dir = 'EmClass/GANConfigs'
base = 'TODO'


if __name__ == '__main__':
    confs = [

        ('128', 15, 'MNIST', 'Gray128',
         f'{base}OutlierTrain{sep}MNIST{sep}run_025{sep}epoch_9{sep}train_ssim.csv',
         f'{base}Reproducibility{sep}MNIST{sep}run_128{sep}epoch_15{sep}ImagesCorrect'
         ),
    ]

    times = []
    errors = []

    run_count = 0

    for (run_nr, epoch, type, dataset, path, generated_path) in confs:
        conf = (
            run_nr,
            root_dir,
            config_dir,
            f'epoch_{epoch}'
        )

        OutlierEvaluationPipeline().run(
            conf,
            path_to_dataset='TODO',
            path_to_train_ssim_distances=path,
            path_to_output_folder=f'{base}Outlier{sep}{type}',
            path_to_correct_generated_images=generated_path,
        )

