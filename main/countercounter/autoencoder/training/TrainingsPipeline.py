from main.countercounter.autoencoder.training.SetupConfigurator import SetupConfigurator
from main.countercounter.autoencoder.training.Training import Training
from main.countercounter.classifier.training.ConfigLoader import ConfigLoader
from main.countercounter.classifier.training.FilePreparation import FilePreparation
from main.countercounter.classifier.training.ModelLoader import ModelLoader


class TrainingsPipeline:

    def __init__(self):
        pass

    def run(self, config_nr, root_dir, config_dir, path_to_dataset=None):
        config = ConfigLoader(root_dir, config_dir).load(config_nr)
        setup = SetupConfigurator(config, root_dir, config_nr, path_to_dataset=path_to_dataset).configure()

        FilePreparation(setup).prepare()
        ModelLoader(setup, config).load_from_config()

        Training(setup).train()