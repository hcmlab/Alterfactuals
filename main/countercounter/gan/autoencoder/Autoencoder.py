import torch
from torch import nn

from main.countercounter.autoencoder.networks.AE_networks import AEV2Net, AEDhurandhar
from main.countercounter.gan.autoencoder.AutoencoderLoss import AutoencoderLoss
from main.countercounter.gan.utils.AbstractTraining import DEVICE


class Autoencoder(nn.Module):

    def __init__(self, path_to_pretrained_model, device, size, loss: AutoencoderLoss, in_channels=1) -> None:
        super().__init__()
        self.path = path_to_pretrained_model
        self.device = device

        self.loss = loss
        self.size = size

        self.in_channels = in_channels
        self._load_model()

    def _load_model(self):
        if self.size == 0:
            self.model = AEV2Net(in_channels=self.in_channels).to(DEVICE)
        elif self.size == 0:
            self.model = AEDhurandhar(in_channels=self.in_channels).to(DEVICE)
        else:
            raise ValueError(f'Unknown model type')

        location = torch.device('cpu') if not torch.cuda.is_available() else torch.device('cuda:0')
        checkpoint = torch.load(self.path, map_location=location)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

    def forward(self, modified_image):
        with torch.no_grad():
            modified_reconstructed = self.model(modified_image)
        return self.loss(modified_image, modified_reconstructed)
