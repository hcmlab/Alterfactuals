from torch import nn


class AutoencoderLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, original, modified):
        euclidean_distance = ((original - modified) ** 2)
        return euclidean_distance.mean()

