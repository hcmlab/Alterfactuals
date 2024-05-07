import torch
import torch.nn as nn

from main.countercounter.csvae.networks.no_variation_models import EncoderXYToWNoVarConv


class CSAE(nn.Module):

    def __init__(self, path_to_pretrained_model, device, n_classes, in_channels=1) -> None:
        super().__init__()
        self.path = path_to_pretrained_model
        self.device = device

        self.n_classes = n_classes

        self.in_channels = in_channels
        self.encoder_xy_to_w = None

        self.loss = nn.MSELoss()

        if path_to_pretrained_model is not None:
            self._load_model()

    def _load_model(self):
        self.encoder_xy_to_w = EncoderXYToWNoVarConv(128, self.n_classes).to(self.device)

        checkpoint = torch.load(self.path, map_location=torch.device('cpu'))  # map_location=torch.device('cuda:0')

        self.encoder_xy_to_w.load_state_dict(checkpoint['encoder_xy_to_w_csvae_state_dict'])

        self.encoder_xy_to_w.eval()

    def forward(self, real_x, real_label, generated_x, generated_label):
        if self.encoder_xy_to_w is None:
            return torch.tensor(0.)

        with torch.no_grad():
            w_real = self.encoder_xy_to_w(real_x, real_label)
            w_generated = self.encoder_xy_to_w(generated_x, generated_label)

        loss = self.loss(w_generated, w_real)
        return loss