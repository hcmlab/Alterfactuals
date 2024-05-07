import torch

from main.countercounter.csvae.training.Setup import Setup
from main.countercounter.csvae.training.CSVAE_loss import CSVAELoss


class CSVAE:

    def __init__(self, setup: Setup, device):
        self.setup = setup

        self.encoder_x_to_z = setup.encoder_x_to_z
        self.encoder_xy_to_w = setup.encoder_xy_to_w
        self.encoder_y_to_w = setup.encoder_y_to_w
        self.decoder_zw_to_x = setup.decoder_zw_to_x
        self.decoder_z_to_y = setup.decoder_z_to_y

        self.device = device

    def get_losses(self, x, y, y_scores):
        w, w_log_var, w_mu, x_pred, y_pred_main, y_pred_delta, z, z_log_var, z_mu, w_mu_prior, w_log_var_prior, w_prior = self._get_values(x, y)

        return CSVAELoss(self.setup).calculate(w_mu, w_log_var, w, z_mu, z_log_var, z, x_pred, x, y_pred_main, y_pred_delta, y, w_mu_prior, w_log_var_prior, w_prior, y_scores)

    def _get_values(self, x, y):
        w_mu, w_log_var, w = self.encoder_xy_to_w(x, y)
        w_mu_prior, w_log_var_prior, w_prior = self.encoder_y_to_w(y)

        z_mu, z_log_var, z = self.encoder_x_to_z(x)

        x_pred = self.decoder_zw_to_x(z, w)

        y_pred_main = self.decoder_z_to_y(z)
        y_pred_delta = self.decoder_z_to_y(z.detach().clone().to(self.device))

        assert torch.all(torch.eq(y_pred_main, y_pred_delta)).item()

        return w, w_log_var, w_mu, x_pred, y_pred_main, y_pred_delta, z, z_log_var, z_mu, w_mu_prior, w_log_var_prior, w_prior

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
        w, w_log_var, w_mu, x_pred, y_pred_main, y_pred_delta, z, z_log_var, z_mu, w_mu_prior, w_log_var_prior, w_prior = self._get_values(x, y)
        CSVAELoss(self.setup).calculate(w_mu, w_log_var, w, z_mu, z_log_var, z, x_pred, x, y_pred_main, y_pred_delta, y,
                                        w_mu_prior, w_log_var_prior, w_prior, y_scores)

        return x_pred, y_pred_delta

    def get_w_z(self, x, y):
        w_mu, w_log_var, w = self.encoder_xy_to_w(x, y)
        z_mu, z_log_var, z = self.encoder_x_to_z(x)
        return w, z