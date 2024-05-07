import torch

from main.countercounter.csvae.networks.CSVAE import CSVAE
from main.countercounter.csvae.training.CSAE_loss import CSAELoss
from main.countercounter.csvae.training.Setup import Setup
from main.countercounter.csvae.training.CSVAE_loss import CSVAELoss


class CSAE(CSVAE):

    def __init__(self, setup: Setup, device):
        super().__init__(setup, device)

    def get_losses(self, x, y, y_scores):
        w, x_pred, y_pred_main, y_pred_delta, z = self._get_values(x, y)

        return CSAELoss(self.setup).calculate(w, z, x_pred, x, y_pred_main, y_pred_delta, y, y_scores)

    def _get_values(self, x, y):
        w = self.encoder_xy_to_w(x, y)

        z = self.encoder_x_to_z(x)

        x_pred = self.decoder_zw_to_x(z, w)

        y_pred_main = self.decoder_z_to_y(z)
        y_pred_delta = self.decoder_z_to_y(z.detach().clone().to(self.device))

        assert torch.all(torch.eq(y_pred_main, y_pred_delta)).item()

        return w, x_pred, y_pred_main, y_pred_delta, z

    def train(self):
        self.encoder_x_to_z.train()
        self.encoder_xy_to_w.train()
        self.decoder_zw_to_x.train()
        self.decoder_z_to_y.train()
        self.encoder_y_to_w.train()

    def eval(self):
        self.encoder_x_to_z.eval()
        self.encoder_xy_to_w.eval()
        self.decoder_zw_to_x.eval()
        self.decoder_z_to_y.eval()
        self.encoder_y_to_w.eval()

    def model_checkpoint_info(self):
        return [
            ('encoder_x_to_z', self.encoder_x_to_z.state_dict()),
            ('encoder_xy_to_w', self.encoder_xy_to_w.state_dict()),
            ('decoder_zw_to_x', self.decoder_zw_to_x.state_dict()),
            ('decoder_z_to_y', self.decoder_z_to_y.state_dict()),
            ('encoder_y_to_w', self.encoder_y_to_w.state_dict()),
        ]

    def get_x_pred_y_pred(self, x, y, y_scores):
        w, x_pred, y_pred_main, y_pred_delta, z = self._get_values(x, y)
        CSAELoss(self.setup).calculate(w, z, x_pred, x, y_pred_main, y_pred_delta, y, y_scores)

        return x_pred, y_pred_delta

    def get_w_z(self, x, y):
        w = self.encoder_xy_to_w(x, y)
        z = self.encoder_x_to_z(x)
        return w, z