from main.countercounter.classifier.evaluation.TrainDistributionEvaluator import TrainDistributionEvaluator
from main.countercounter.classifier.training.ConfigLoader import ConfigLoader
from main.countercounter.classifier.training.FilePreparation import FilePreparation
from main.countercounter.classifier.training.ModelLoader import ModelLoader
from main.countercounter.classifier.training.SetupConfigurator import SetupConfigurator


class TrainActivationEvaluationPipeline:

    def run(self, config_nr, root_dir, config_dir, checkpoint, path_to_dataset=None, path_to_output_folder=None):
        config = ConfigLoader(root_dir, config_dir).load(config_nr)
        setup = SetupConfigurator(config, root_dir, config_nr, eval=True, path_to_dataset=path_to_dataset, add_identifier=True).configure()

        ModelLoader(setup).load(checkpoint)
        path_to_output_folder = FilePreparation(setup).make_output_dir_for_run(path_to_output_folder, config_nr, checkpoint)
        TrainDistributionEvaluator(setup, path_to_output_folder).evaluate()