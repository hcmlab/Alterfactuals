import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from main.countercounter.gan._execution.evaluation.reproducibility.EvaluationPreparationPipeline import \
    EvaluationPreparationPipeline


root_dir = 'TODO'
config_dir = 'EmClass/GANConfigs'


if __name__ == '__main__':
    confs = [

        ('128', 15, 'MNIST', 'Gray128'),
    ]

    times = []
    errors = []

    run_count = 0

    for (run_nr, epoch, type, dataset) in confs:
        conf = (
            run_nr,
            root_dir,
            config_dir,
            f'epoch_{epoch}'
        )

        EvaluationPreparationPipeline().run(
            conf,
            path_to_dataset='TODO',
            path_to_output_folder='TODO',
        )

