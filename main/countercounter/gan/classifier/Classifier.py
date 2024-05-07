import torch
from torch import nn
import pandas as pd

from main.countercounter.classifier.emotion_classifier.CustomNets import CustomNet, CustomNetSmall, \
    CustomNetSmallBinary, CustomNetBinary, CustomNetExtremelySmallBinary, CustomNetSmallLogits, SoftmaxWrapper, \
    CustomNetSmallGAPLogits, SoftmaxLogitWrapper, CustomNetSmallNoGAPLogits, ModifiedAlexNet
from main.countercounter.classifier.emotion_classifier.ResnetWrapper import ResnetWrapper
from main.countercounter.gan.classifier.ClassifierLoss import ClassifierLoss
from main.countercounter.gan.utils.AbstractTraining import DEVICE


class Classifier(nn.Module):

    def __init__(self, path_to_pretrained_model, device, size, loss: ClassifierLoss, n_classes, in_channels=1) -> None:
        super().__init__()
        self.path = path_to_pretrained_model
        self.device = device

        self.n_classes = n_classes

        self.loss = loss

        self.size = size

        self.in_channels = in_channels
        self._load_model()

    def _load_model(self):
        if self.size == 18:
            self.model = ResnetWrapper(n_classes=self.n_classes, size=18, transfer_learning=False, grayscale=True, use_softmax=False).to(DEVICE)
        elif self.size == 0:
            self.model = CustomNetSmallBinary(self.in_channels).to(self.device)
        elif self.size == 3:
            self.model = CustomNetBinary(self.in_channels).to(self.device)
        elif self.size == -1:
            self.model = CustomNetExtremelySmallBinary(self.in_channels).to(self.device)
        elif self.size == 8:
            inner_model = CustomNetSmallLogits(n_classes=self.n_classes).to(DEVICE)
            self.model = SoftmaxWrapper(inner_model).to(DEVICE)
        elif self.size == 9:
            self.inner_model = CustomNetSmallGAPLogits(n_classes=self.n_classes).to(DEVICE)
            self.model = SoftmaxLogitWrapper(self.inner_model).to(DEVICE)
        elif self.size == 10:
            self.inner_model = CustomNetSmallGAPLogits(n_classes=self.n_classes, in_channels=3).to(DEVICE)
            self.model = SoftmaxLogitWrapper(self.inner_model).to(DEVICE)
        elif self.size == 11:
            self.inner_model = CustomNetSmallNoGAPLogits(n_classes=self.n_classes).to(DEVICE)
            self.model = SoftmaxLogitWrapper(self.inner_model).to(DEVICE)
        elif self.size == 12:
            self.inner_model = ModifiedAlexNet(n_classes=self.n_classes).to(DEVICE)
            self.model = SoftmaxLogitWrapper(self.inner_model).to(DEVICE)
        else:
            raise ValueError(f'Unknown model type')

        checkpoint = torch.load(self.path, map_location=torch.device('cpu'))#  map_location=torch.device('cuda:0')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        # self.model.use_softmax = True   # to ensure that softmax is used to normalize the outputs
        self.model.eval()

    def forward(self, original_image, modified_image):
        # original_image = normalize_for_resnet(original_image)
        # modified_image = normalize_for_resnet(modified_image)

        with torch.no_grad():
            output1 = self.model(original_image)
            output2 = self.model(modified_image)
        return self.loss(output1, output2)

    def get_class(self, images):
        with torch.no_grad():
            preds = self.model(images)

        if preds.size(1) > 1:
            _, classes = torch.max(preds, 1)
        else:
            classes = (preds >= 0.5).int()

        return classes

    def get_class_and_logits(self, images):
        with torch.no_grad():
            preds = self.model(images)

        if preds.size(1) > 1:
            _, classes = torch.max(preds, 1)
        else:
            classes = (preds >= 0.5).int()

        return classes, preds

    def pred(self, images):
        with torch.no_grad():
            preds = self.model(images)

        return preds

    def pred_with_grad(self, images):
        preds = self.model(images)
        return preds

    def get_activations(self, original, modified):
        acts_original = self._get_acts(original)
        acts_modified = self._get_acts(modified)

        return acts_original, acts_modified

    def _get_acts(self, image):
        with torch.no_grad():
            output = self.inner_model(image)

        acts = output[1][0].cpu().detach().numpy()
        acts = acts.tolist()

        pred = self.get_class(image)

        acts.insert(0, pred[0].item())

        columns = ['Class'] + [f'feature_{idx}' for idx in range(len(acts[1:]))]
        df = pd.DataFrame([acts], columns=columns)
        return df


