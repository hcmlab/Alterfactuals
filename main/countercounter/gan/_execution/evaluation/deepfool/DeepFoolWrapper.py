from torch import nn

from main.countercounter.classifier.emotion_classifier.CustomNets import SoftmaxWrapper


class DeepFoolWrapper(nn.Module):

    def __init__(self, classifier):
        super().__init__()

        if isinstance(classifier.model, SoftmaxWrapper):
            self.classifier = classifier.model.model
        else:
            raise NotImplementedError()

    def forward(self, x):
        return self.classifier(x)