from main.countercounter.classifier.training.ConfigLoader import ConfigLoader
from main.countercounter.classifier.training.FilePreparation import FilePreparation
from main.countercounter.classifier.training.ModelLoader import ModelLoader
from main.countercounter.classifier.training.SetupConfigurator import SetupConfigurator
from main.countercounter.classifier.training.Training import Training


class TrainingsPipeline:

    def __init__(self):
        pass

    def run(self, config_nr, root_dir, config_dir, path_to_dataset=None):
        config = ConfigLoader(root_dir, config_dir).load(config_nr)
        setup = SetupConfigurator(config, root_dir, config_nr, path_to_dataset=path_to_dataset).configure()

        FilePreparation(setup).prepare()
        ModelLoader(setup, config).load_from_config()

        Training(setup).train()