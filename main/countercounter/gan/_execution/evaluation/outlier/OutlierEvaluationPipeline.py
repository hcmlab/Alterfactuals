from main.countercounter.classifier.training.FilePreparation import FilePreparation
from main.countercounter.gan._execution.evaluation.ModelLoader import ModelLoader
from main.countercounter.gan._execution.evaluation.outlier.OutlierEvaluator import OutlierEvaluator
from main.countercounter.gan._execution.execution_utils.ConfigLoader import ConfigLoader
from main.countercounter.gan._execution.execution_utils.SetupConfigurator import SetupConfigurator


class OutlierEvaluationPipeline:

    def run(self, conf, path_to_dataset, path_to_train_ssim_distances, path_to_output_folder, path_to_correct_generated_images):
        config_nr, root_dir, config_dir, checkpoint = conf

        config = ConfigLoader(root_dir, config_dir).load(config_nr)
        setup = SetupConfigurator(config, root_dir, config_nr, eval=True, path_to_dataset=path_to_dataset, add_identifier=True).configure()

        setup_for_generated_images = SetupConfigurator(config, root_dir, config_nr, eval=True, path_to_dataset=path_to_correct_generated_images, add_identifier=True, size_check=False).configure()

        path = FilePreparation(setup).make_output_dir_for_run(path_to_output_folder, config_nr, checkpoint)
        ModelLoader(setup).load(checkpoint)
        OutlierEvaluator(
            setup,
            setup_for_generated_images,
            path_to_output_folder=path,
            path_to_train_ssim_distances=path_to_train_ssim_distances,
        ).evaluate()
