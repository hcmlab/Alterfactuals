import os
import shutil
from os.path import sep

from main.countercounter.gan._execution.execution_utils.Setup import Setup


class FilePreparation:

    def __init__(self, setup: Setup):
        self.root_dir = setup.root_dir
        self.checkpoint_dir = setup.checkpoints_dir
        self.img_dir = setup.image_dir
        self.tensorboard_dir = setup.tensorboard_dir
        self.model_dir = setup.model_dir

        self.config_nr = setup.config_nr

        self.setup = setup

    def prepare(self):
        if not self.setup.minimal_logging:
            checkpoints_dir_name = self._make_dir_for_base_dir(self.checkpoint_dir)
            tensorboard_dir_name = self._make_dir_for_base_dir(self.tensorboard_dir)

            self.setup.checkpoints_dir = checkpoints_dir_name
            self.setup.tensorboard_dir = tensorboard_dir_name

        images_dir_name = self._make_dir_for_base_dir(self.img_dir)
        model_dir_name = self._make_dir_for_base_dir(self.model_dir)

        self.setup.image_dir = images_dir_name
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

    def prepare_dir_for_generated_images(self, path_to_generated_images_dir, epoch):
        full_path = f'{path_to_generated_images_dir}{sep}run_{self.config_nr}{sep}{epoch}'

        self._make_dir(full_path)

        return full_path

    def prepare_dir_for_example_images(self, path_to_generated_images_dir, run1, run2, epoch1, epoch2):
        full_path = f'{path_to_generated_images_dir}{sep}runs_{run1}_{run2}{sep}epochs_{epoch1}_{epoch2}'

        self._make_dir(full_path)

        return full_path