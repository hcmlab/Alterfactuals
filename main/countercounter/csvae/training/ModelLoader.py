import torch

from main.countercounter.csvae.training.Setup import Setup


class ModelLoader:

    def __init__(self, setup: Setup, config=None):
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


        self.setup.main_optimizer.load_state_dict(checkpoint['main_optimizer_state_dict'])
        self.setup.delta_optimizer.load_state_dict(checkpoint['delta_optimizer_state_dict'])

        self.setup.main_scheduler.load_state_dict(checkpoint['main_lr_scheduler_state_dict'])
        self.setup.delta_scheduler.load_state_dict(checkpoint['delta_lr_scheduler_state_dict'])

        self.setup.encoder_x_to_z.load_state_dict(checkpoint['encoder_x_to_z_csvae_state_dict'])
        self.setup.encoder_xy_to_w.load_state_dict(checkpoint['encoder_xy_to_w_csvae_state_dict'])
        self.setup.encoder_y_to_w.load_state_dict(checkpoint['encoder_y_to_w_csvae_state_dict'])
        self.setup.decoder_zw_to_x.load_state_dict(checkpoint['decoder_zw_to_x_csvae_state_dict'])
        self.setup.decoder_z_to_y.load_state_dict(checkpoint['decoder_z_to_y_csvae_state_dict'])

    def load_from_config(self):
        continue_training = self.config['continue_training']

        if not continue_training['continue']:
            return

        checkpoint = continue_training['checkpoint']
        run = continue_training['run_to_continue']

        original_checkpoint_dir = self.setup.checkpoints_dir[:self.setup.checkpoints_dir.rfind('/') + 1]

        full_path = original_checkpoint_dir + f'/run_{run}/' + checkpoint

        self._load_from_path(full_path)