from main.countercounter.classifier.training.FilePreparation import FilePreparation
from main.countercounter.gan._execution.evaluation.ModelLoader import ModelLoader
from main.countercounter.gan._execution.evaluation.reproducibility.EvaluationPreparer import EvaluationPreparer
from main.countercounter.gan._execution.execution_utils.ConfigLoader import ConfigLoader
from main.countercounter.gan._execution.execution_utils.SetupConfigurator import SetupConfigurator


class EvaluationPreparationPipeline:

    def run(self, conf, path_to_dataset, path_to_output_folder):
        config_nr, root_dir, config_dir, checkpoint = conf

        config = ConfigLoader(root_dir, config_dir).load(config_nr)
        setup = SetupConfigurator(config, root_dir, config_nr, eval=True, path_to_dataset=path_to_dataset, add_identifier=True, shuffle=False).configure()

        path = FilePreparation(setup).make_output_dir_for_run(path_to_output_folder, config_nr, checkpoint)
        ModelLoader(setup).load(checkpoint)
        EvaluationPreparer(
            setup,
            path_to_output_folder=path,
        ).evaluate()
