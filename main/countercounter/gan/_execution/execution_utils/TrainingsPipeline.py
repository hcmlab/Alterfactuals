from main.countercounter.gan._execution.evaluation.ModelLoader import ModelLoader
from main.countercounter.gan._execution.execution_utils.ConfigLoader import ConfigLoader
from main.countercounter.gan._execution.execution_utils.FilePreparation import FilePreparation
from main.countercounter.gan._execution.execution_utils.SetupConfigurator import SetupConfigurator
from main.countercounter.gan.utils.training import Training


class TrainingsPipeline:

    def __init__(self):
        pass

    def run(self, config_nr, root_dir, config_dir, path_to_dataset=None, path_to_kdes=None):
        config = ConfigLoader(root_dir, config_dir).load(config_nr)
        setup = SetupConfigurator(config, root_dir, config_nr, path_to_dataset=path_to_dataset, path_to_kdes=path_to_kdes).configure()

        FilePreparation(setup).prepare()
        ModelLoader(setup, config).load_from_config()

        Training(setup).train()
