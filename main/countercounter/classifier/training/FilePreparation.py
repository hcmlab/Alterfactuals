import os
import shutil
from os.path import sep

from main.countercounter.classifier.training.Setup import Setup


class FilePreparation:

    def __init__(self, setup: Setup):
        self.root_dir = setup.root_dir
        self.checkpoint_dir = setup.checkpoints_dir
        self.tensorboard_dir = setup.tensorboard_dir
        self.model_dir = setup.model_dir

        self.config_nr = setup.config_nr

        self.setup = setup

    def prepare(self):
        checkpoints_dir_name = self._make_dir_for_base_dir(self.checkpoint_dir)
        tensorboard_dir_name = self._make_dir_for_base_dir(self.tensorboard_dir)
        model_dir_name = self._make_dir_for_base_dir(self.model_dir)

        self.setup.checkpoints_dir = checkpoints_dir_name
        self.setup.tensorboard_dir = tensorboard_dir_name
        self.setup.model_dir = model_dir_name

    def _make_dir_for_base_dir(self, base_dir):
        dir_name = self._make_full_path(base_dir)

        self._make_dir(dir_name)
        return dir_name

    def _make_full_path(self, path):
        return path + '/' + "run_" + self.config_nr

    def _make_dir(self, dir_name):
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name, ignore_errors=True)

        os.makedirs(dir_name)

    def make_output_dir_for_run(self, base_dir, config_nr, checkpoint):
        path = f'{base_dir}{sep}run_{config_nr}{sep}{checkpoint}'
        self._make_dir(path)
        return path

    def make_output_dir_for_type(self, base_dir, type):
        path = f'{base_dir}{sep}{type}'
        self._make_dir(path)
        return path