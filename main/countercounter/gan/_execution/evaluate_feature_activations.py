import os
from multiprocessing.pool import Pool
from os.path import sep

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


from main.countercounter.gan._execution.evaluation.activationdistribution.ActivationEvaluationPipeline import \
    ActivationEvaluationPipeline


root_dir = 'TODO'
config_dir = 'EmClass/GANConfigs'

evaluation_dir = 'TODO'


def evaluate(from_csv_confs):
    for (
    config_nr_for_classifier, dataset, type, is_benchmark, act_csv_path1, act_csv_path2, act_csv, dist_models, svm_path,
    csv_logit_path_1, csv_logit_path_2, path_to_logit_csv, path_to_logit_models) in from_csv_confs:

        try:

            ActivationEvaluationPipeline().run_from_csv(
                root_dir=root_dir,
                config_dir=config_dir,
                config_nr_for_classifier=config_nr_for_classifier,
                type=type,
                path_csv_1=act_csv_path1,
                path_csv_2=act_csv_path2,
                path_to_output_folder=f'{evaluation_dir}Activation{sep}{dataset}{sep}{"Benchmark" if is_benchmark else config_nr_for_classifier}{sep}{type}',
                path_to_distribution_csv=act_csv,
                path_to_logit_csv=path_to_logit_csv,
                path_to_distribution_models=dist_models,
                path_to_logit_models=path_to_logit_models,
                svm_path=svm_path,
                csv_logit_path_1=csv_logit_path_1,
                csv_logit_path_2=csv_logit_path_2,
            )

        except Exception as e:
            print(f'---------------- ERROR: {e}')

            raise e


if __name__ == '__main__':

    from_csv_confs_2 = [

        ('132', 'FashionMNIST', '74', False,
         f'{evaluation_dir}Reproducibility{sep}FashionMNIST{sep}run_132{sep}epoch_74{sep}TEST{sep}Alter Original Activations.csv',
         f'{evaluation_dir}Reproducibility{sep}FashionMNIST{sep}run_132{sep}epoch_74{sep}TEST{sep}Alter Generated Activations.csv',
         f'{evaluation_dir}FeatureDistribution{sep}FashionMNIST{sep}run_026{sep}epoch_40{sep}activations_train.csv',
         f'{evaluation_dir}FeatureDistribution{sep}FashionMNIST{sep}run_026{sep}epoch_40{sep}best_models.csv',
         f'{evaluation_dir}ActivationSVM{sep}FashionMNIST{sep}run_026{sep}epoch_40{sep}svm.joblib',
         f'{evaluation_dir}Reproducibility{sep}FashionMNIST{sep}run_132{sep}epoch_74{sep}TEST{sep}Alter Original Logits.csv',
         f'{evaluation_dir}Reproducibility{sep}FashionMNIST{sep}run_132{sep}epoch_74{sep}TEST{sep}Alter Generated Logits.csv',
         f'{evaluation_dir}FeatureDistribution{sep}FashionMNIST{sep}run_026{sep}epoch_40{sep}logits_train.csv',
         f'{evaluation_dir}FeatureDistribution{sep}FashionMNIST{sep}run_026{sep}epoch_40{sep}logits{sep}best_logit_models.csv',
         ),
    ]

    confs = [from_csv_confs_2]

    pool = Pool(processes=2)
    pool.map(evaluate, confs)
