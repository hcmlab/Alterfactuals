import torch
from main.countercounter.classifier.emotion_classifier.CustomNets import CustomNetSmallGAPLogits

# source: Keane Semifactuals
from main.countercounter.gan.utils.AbstractTraining import DEVICE


def return_feature_contribution_data(data_loader, cnn: CustomNetSmallGAPLogits, num_classes=2):
    #full_data = dict()
    pred_idx = dict()
    logit_idx = dict()

    for class_name in list(range(num_classes)):
        pred_idx[class_name] = list()
        logit_idx[class_name] = list()

    for i, data in enumerate(data_loader):
        # print progress
        if i % 500 == 0:
            print(100 * round(i / len(data_loader), 2), "% complete...")

        image, _ = data
        # ckaarle: added
        if image.size(0) > 1:
            raise NotImplementedError

        # ckaarle: added
        image = image.to(DEVICE)

        #label = int(label.detach().numpy())
        output = cnn(image)
        acts = output[1][0].detach().numpy()
        logits = output[0][0].detach().numpy()

        # ckaarle: added softmax just to be consistent
        softmax_preds = torch.softmax(cnn(image)[0], dim=1)
        pred = int(torch.argmax(softmax_preds).detach().numpy())
        pred_idx[pred].append(acts.tolist())

        logits_list = logits.tolist()
        logit_idx[pred].append(logits_list)

    return pred_idx, logit_idx


def return_feature_contribution_data_with_identifier(data_loader, cnn: CustomNetSmallGAPLogits, num_classes=2):
    #full_data = dict()
    pred_idx = dict()
    logit_idx = dict()

    for class_name in list(range(num_classes)):
        pred_idx[class_name] = list()
        logit_idx[class_name] = list()

    for i, data in enumerate(data_loader):
        # print progress
        if i % 500 == 0:
            print(100 * round(i / len(data_loader), 2), "% complete...")

        image, _, filename = data
        # ckaarle: added
        if image.size(0) > 1:
            raise NotImplementedError

        # ckaarle: added
        image = image.to(DEVICE)

        #label = int(label.detach().numpy())
        output = cnn(image)
        acts = output[1][0].detach().numpy()
        logits = output[0][0].detach().numpy()

        # ckaarle: added softmax just to be consistent
        softmax_preds = torch.softmax(cnn(image)[0], dim=1)
        pred = int(torch.argmax(softmax_preds).detach().numpy())

        act_list = acts.tolist()
        act_list.append(filename[0])  # tuple for some reason
        pred_idx[pred].append(act_list)

        logits_list = logits.tolist()
        logits_list.append(filename[0])
        logit_idx[pred].append(logits_list)

    return pred_idx, logit_idx