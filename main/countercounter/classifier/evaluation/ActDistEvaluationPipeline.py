from main.countercounter.classifier.evaluation.ActDistWeightEvaluator import ActDistWeightEvaluator
from main.countercounter.classifier.training.ConfigLoader import ConfigLoader
from main.countercounter.classifier.training.FilePreparation import FilePreparation
from main.countercounter.classifier.training.ModelLoader import ModelLoader
from main.countercounter.classifier.training.SetupConfigurator import SetupConfigurator


class ActDistEvaluationPipeline:

    def run(self, config_nr, root_dir, config_dir, checkpoint, path_to_dataset=None, path_to_output_folder=None, activation_csv=None):
        config = ConfigLoader(root_dir, config_dir).load(config_nr)
        setup = SetupConfigurator(config, root_dir, config_nr, eval=True, path_to_dataset=path_to_dataset, add_identifier=False).configure()

        ModelLoader(setup).load(checkpoint)
        path_to_output_folder = FilePreparation(setup).make_output_dir_for_run(path_to_output_folder, config_nr, checkpoint)

        ActDistWeightEvaluator(setup, path_to_output_folder, activation_csv).evaluate()