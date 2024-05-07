from main.countercounter.classifier.training.FilePreparation import FilePreparation
from main.countercounter.gan._execution.evaluation.ModelLoader import ModelLoader
from main.countercounter.gan._execution.evaluation.activationdistribution.ActivationEvaluator import ActivationEvaluator
from main.countercounter.gan._execution.execution_utils.ConfigLoader import ConfigLoader
from main.countercounter.gan._execution.execution_utils.SetupConfigurator import SetupConfigurator


class ActivationEvaluationPipeline:

    def run(self, conf, path_to_dataset, path_to_output_folder, path_to_distribution_csv, path_to_logit_csv, path_to_distribution_models, path_to_logit_models, svm_path):
        config_nr, root_dir, config_dir, checkpoint = conf

        config = ConfigLoader(root_dir, config_dir).load(config_nr)
        setup = SetupConfigurator(config, root_dir, config_nr, eval=True, path_to_dataset=path_to_dataset).configure()

        path = FilePreparation(setup).make_output_dir_for_run(path_to_output_folder, config_nr, checkpoint)
        ModelLoader(setup).load(checkpoint)
        ActivationEvaluator(
            setup,
            path_to_output_folder=path,
            path_to_distribution_csv=path_to_distribution_csv,
            path_to_logit_csv=path_to_logit_csv,
            path_to_distribution_models=path_to_distribution_models,
            path_to_logit_models=path_to_logit_models,
            svm_path=svm_path,
        ).evaluate()

    def run_from_csv(
            self,
            root_dir,
            config_dir,
            config_nr_for_classifier,
            type,
            path_csv_1,
            path_csv_2,
            path_to_output_folder,
            path_to_distribution_csv,
            path_to_logit_csv,
            path_to_distribution_models,
            path_to_logit_models,
            svm_path,
            csv_logit_path_1,
            csv_logit_path_2,
    ):
        config = ConfigLoader(root_dir, config_dir).load(config_nr_for_classifier)
        setup = SetupConfigurator(config, root_dir, config_nr_for_classifier, eval=True, path_to_dataset=None).configure_minimal()

        path = FilePreparation(setup).make_output_dir_for_type(path_to_output_folder, type)
        ActivationEvaluator(
            setup,
            path_to_output_folder=path,
            path_to_distribution_csv=path_to_distribution_csv,
            path_to_logit_csv=path_to_logit_csv,
            path_to_distribution_models=path_to_distribution_models,
            path_to_logit_models=path_to_logit_models,
            svm_path=svm_path,
            csv_act_path1=path_csv_1,
            csv_act_path2=path_csv_2,
            csv_logit_path_1=csv_logit_path_1,
            csv_logit_path_2=csv_logit_path_2,
        ).evaluate()