import torch
from torch import nn
import torch.nn.functional as F


class ClassifierLoss(nn.Module):
    pass


class MSELoss(ClassifierLoss):

    def __init__(self, class_to_generate):
        super().__init__()
        self.loss = nn.MSELoss()

    def forward(self, original, modified):
        output1 = self.model(original)
        output2 = self.model(modified)

        output1_normalized = F.normalize(output1)
        output2_normalized = F.normalize(output2)

        mse = self.loss(output1_normalized, output2_normalized)

        assert mse >= 0.
        assert mse <= 1.

        return mse


class InvertedMSELoss(ClassifierLoss):

    def __init__(self, class_to_generate):
        super().__init__()

        self.loss = MSELoss(class_to_generate)

    def forward(self, original, modified):
        mse = self.loss(original, modified)

        return 1 - mse


class ArgmaxLoss(ClassifierLoss):

    def __init__(self, class_to_generate):
        super().__init__()

    def forward(self, original, modified):
        return torch.mean(self._distance(original, modified))

    def _distance(self, original, modified):
        originalMod = (original >= 0.5).float()

        # original >= 0.5 --> goal is 1 --> loss is 1 - current value
        # original < 0.5 --> goal is 0 --> loss is current value

        loss_to_achieve_0 = modified
        loss_to_achieve_1 = 1 - modified

        loss = (1 - originalMod) * loss_to_achieve_0 + (originalMod) * loss_to_achieve_1
        return loss


class InvertedArgmaxLoss(ClassifierLoss):

    def __init__(self, class_to_generate=None):
        super().__init__()
        self.class_to_generate = class_to_generate

    def forward(self, original, modified):
        return torch.mean(self._distance(original, modified))

    def _distance(self, original, modified):
        # input is a float value after sigmoid, so [0,1]

        # if original >= 0.5 --> goal is 0 --> loss is current value
        # if original < 0.5 --> goal is 1 --> loss is 1 - current value

        originalMod = (original >= 0.5).float()
        loss_to_achieve_0 = modified
        loss_to_achieve_1 = 1 - modified

        if self.class_to_generate is not None:
            if self.class_to_generate == 0:
                loss = loss_to_achieve_0
            elif self.class_to_generate == 1:
                loss = loss_to_achieve_1
            else:
                raise ValueError(f'Unknown class to generate: {self.class_to_generate}')
        else:
            loss = (originalMod * loss_to_achieve_0) + ((1-originalMod) * loss_to_achieve_1)

        return loss


class BCELossWrapper(ClassifierLoss):

    def __init__(self, class_to_generate):
        super().__init__()
        self.loss = nn.BCELoss()

    def forward(self, original, modified):
        target_class = (original >= 0.5).float()
        return self.loss(modified, target_class)


class InvertedBCELossWrapper(ClassifierLoss):

    def __init__(self, class_to_generate):
        super().__init__()
        self.loss = nn.BCELoss()

    def forward(self, original, modified):
        target_class = (original < 0.5).float() #  invert class label
        return self.loss(modified, target_class)


class AbsDistLoss(ClassifierLoss):

    def __init__(self, class_to_generate):
        super().__init__()

    def forward(self, original, modified):
        return torch.mean(torch.abs(original - modified))


class InvertedAbsDistLoss(ClassifierLoss):

    def __init__(self, class_to_generate):
        super().__init__()

    def forward(self, original, modified):
        pass # TODO