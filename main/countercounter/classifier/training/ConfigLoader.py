import yaml


class ConfigLoader:

    def __init__(self, root_dir, config_dir):
        self.root_dir = root_dir
        self.config_dir = config_dir

    def load(self, config_nr):
        config_path = self.root_dir + '/' + self.config_dir + '/' + f'config_{config_nr}.yml'

        with open(config_path) as f:
            return yaml.load(f)