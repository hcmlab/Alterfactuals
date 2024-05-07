import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from main.countercounter.classifier.evaluation.TrainActivationEvaluationPipeline import \
    TrainActivationEvaluationPipeline

root_dir = 'TODO'
config_dir = 'EmClass/configs'


if __name__ == '__main__':
    confs = [
        (
            '025',
            9,
            'TODO',
            'TODO',
        ),
    ]

    for conf in confs:
        config_nr, epoch, dataset_path, output_folder = conf

        TrainActivationEvaluationPipeline().run(
            config_nr,
            root_dir,
            config_dir,
            f'epoch_{epoch}',
            path_to_dataset=dataset_path,
            path_to_output_folder=output_folder,
        )
        print('-----------------')