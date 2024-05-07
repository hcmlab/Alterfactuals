from main.countercounter.csvae.training.CSVAE_loss import CSVAELoss
from main.countercounter.csvae.training.Setup import Setup


class CSAELoss(CSVAELoss):

    def __init__(self, setup: Setup):
        super().__init__(setup)

    def calculate(self, w, z, x_pred, x, y_pred_main, y_pred_delta, y, y_scores):
        main_loss, recon_loss, m2 = self._main_loss(x, x_pred, y_pred_main, y, z, w, y_pred_delta, y_scores)
        delta_loss = self._delta_loss(y, y_pred_delta, y_scores)
        return main_loss, delta_loss, recon_loss, m2

    def _main_loss(self, x, x_pred, y_pred_main, y, z, w, y_pred_delta, y_scores):
        recon_loss = self._x_recon_loss(x, x_pred)
        m2 = self._m2(y_pred_main)

        main_loss = \
            (self.beta1 * recon_loss + self.beta4 * m2).mean()

        return main_loss, recon_loss, m2

    def _x_recon_loss(self, x, x_pred):
        loss = self._x_recon_loss_function
        return loss(x_pred, x).mean()