import torch
import torch.distributions as dists
import torch.nn as nn
import numpy as np

from main.countercounter.csvae.training.Setup import Setup
from main.countercounter.gan.utils.AbstractTraining import DEVICE


class CSVAELoss:

    def __init__(self, setup: Setup):
        self.beta1 = setup.beta1
        self.beta2 = setup.beta2
        self.beta3 = setup.beta3
        self.beta4 = setup.beta4
        self.beta5 = setup.beta5

        self._x_recon_loss_function = setup.x_recon_loss_function
        self.z_dim = setup.z_size
        self.device = DEVICE

    def calculate(self, w_mu, w_log_var, w, z_mu, z_log_var, z, x_pred, x, y_pred_main, y_pred_delta, y, w_mu_prior, w_log_var_prior, w_prior, y_scores):
        main_loss = self._main_loss(x, x_pred, w_mu, w_log_var, w_mu_prior, w_log_var_prior, z_mu, z_log_var, y_pred_main, y)
        delta_loss = self._delta_loss(y, y_pred_delta, y_scores)
        return main_loss, delta_loss

    def _main_loss(self, x, x_pred, w_mu, w_log_var, w_mu_prior, w_log_var_prior, z_mu, z_log_var, y_pred_main, y):
        recon_loss = self._x_recon_loss(x, x_pred)
        kl_w = self._kl_w_loss(w_mu, w_log_var, w_mu_prior, w_log_var_prior, y)
        kl_z = self._kl_z_loss(z_mu, z_log_var)
        m2 = self._m2(y_pred_main)

        main_loss = \
            (self.beta1 * recon_loss + \
            self.beta2 * kl_w + \
            self.beta3 * kl_z + \
            self.beta4 * m2).mean()

        return main_loss

    def _x_recon_loss(self, x, x_pred):
        loss = self._x_recon_loss_function
        return loss(x_pred, x)

    def _kl_w_loss(self, w_mu, w_log_var, w_mu_prior, w_log_var_prior, y):
        """
        The paper defines two distinct normal distributions for y == 1 and y == 0.
        The distribution for y == 1 has mu = (0, 0) and sigma = (0.1, 0.1).
        The distribution for y == 0 has mu = (3, 3) and sigma = (1, 1).
        """
        kl1 = self._kl(w_mu, w_log_var, torch.zeros_like(w_mu), torch.zeros_like(w_log_var) + np.log(0.01))
        kl0 = self._kl(w_mu, w_log_var, torch.ones_like(w_mu) * 3., torch.ones_like(w_log_var))

        return torch.where(y == 1, kl1, kl0).mean()

    def _kl_z_loss(self, z_mu, z_log_var):
        # return self._kl(self.z_mu, self.z_log_var, torch.zeros_like(self.z_mu), torch.ones_like(self.z_log_var))
        z_dist = dists.MultivariateNormal(z_mu.flatten(), torch.diag(z_log_var.flatten().exp()).to(self.device))
        z_prior_dist = dists.MultivariateNormal(torch.zeros(self.z_dim * z_mu.size()[0]).to(self.device), torch.eye(self.z_dim * z_mu.size()[0]).to(self.device))

        z_kl = dists.kl.kl_divergence(z_dist, z_prior_dist)
        return z_kl

    def _kl(self, mu1, logvar1, mu2, logvar2):
        # closed form for multivariate gaussian
        std1 = torch.exp(0.5 * logvar1)
        std2 = torch.exp(0.5 * logvar2)
        return torch.sum(torch.log(std2) - torch.log(std1) + 0.5 * (torch.exp(logvar1) + (mu1 - mu2) ** 2) / torch.exp(logvar2) - 0.5, dim=-1)

    def _m2(self, y_pred_main):
        # equivalent to - H(Y|Z)
        return (y_pred_main.log() * y_pred_main).sum()

    def _delta_loss(self, y, y_pred_delta, y_scores):
        return nn.BCELoss()(y_pred_delta, y_scores)
