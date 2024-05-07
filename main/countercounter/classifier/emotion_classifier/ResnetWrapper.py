from torch import nn
import torchvision.models as models


class ResnetWrapper(nn.Module):

    def __init__(self, n_classes, size=50, transfer_learning=True, grayscale=False, use_softmax=True):
        super().__init__()

        self.use_softmax = use_softmax

        if size == 50:
            self.model = models.resnet50(pretrained=transfer_learning)
        elif size == 18:
            self.model = models.resnet18(pretrained=transfer_learning)
        else:
            raise ValueError(f'Unknown model size {size}.')

        #for param in self.model.parameters():
         #   param.requires_grad = False

        if grayscale:
            self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, n_classes)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        y = self.model(x)

        if self.use_softmax:
            y = self.softmax(y)

        return y