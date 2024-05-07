from main.countercounter.autoencoder.evaluation.Evaluator import Evaluator
from main.countercounter.autoencoder.training.SetupConfigurator import SetupConfigurator
from main.countercounter.classifier.training.ConfigLoader import ConfigLoader
from main.countercounter.classifier.training.ModelLoader import ModelLoader


class EvaluationPipeline:

    def run(self, config_nr, root_dir, config_dir, checkpoint, path_to_dataset=None):
        config = ConfigLoader(root_dir, config_dir).load(config_nr)
        setup = SetupConfigurator(config, root_dir, config_nr, eval=True, path_to_dataset=path_to_dataset, add_identifier=False).configure()

        ModelLoader(setup).load(checkpoint)
        Evaluator(setup, config_nr, checkpoint).evaluate()