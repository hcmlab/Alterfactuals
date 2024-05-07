import torch


class ModelLoader:

    def __init__(self, setup, config=None):
        self.setup = setup
        self.config = config

    def load(self, checkpoint):
        full_path = self.setup.checkpoints_dir + f'/run_{self.setup.config_nr}/' + checkpoint

        self._load_from_path(full_path)

    def _load_from_path(self, full_path):
        if not torch.cuda.is_available():
            checkpoint = torch.load(full_path, map_location=torch.device('cpu'))
        else:
            checkpoint = torch.load(full_path, map_location=torch.device('cuda:0'))

        self.setup.generator.load_state_dict(checkpoint['generator_model_state_dict'])
        self.setup.discriminator.load_state_dict(checkpoint['discriminator_model_state_dict'])
        self.setup.generator_optimizer.load_state_dict(checkpoint['generator_optimizer_state_dict'])
        self.setup.discriminator_optimizer.load_state_dict(checkpoint['discriminator_optimizer_state_dict'])

    def load_from_config(self):
        continue_training = self.config['continue_training']

        if not continue_training['continue']:
            return

        checkpoint = continue_training['checkpoint']
        run = continue_training['run_to_continue']

        original_checkpoint_dir = self.setup.checkpoints_dir[:self.setup.checkpoints_dir.rfind('/') + 1]

        full_path = original_checkpoint_dir + f'/run_{run}/' + checkpoint

        self._load_from_path(full_path)