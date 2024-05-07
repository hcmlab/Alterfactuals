import torch
from torch import nn


class CustomNet(nn.Module):

    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes

        self.kernel = 3
        self.stride = 1
        self.padding = 0

        self.pool = 2

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=self.kernel, stride=self.stride, padding=self.padding)
        self.conv1_batch = nn.BatchNorm2d(126)
        self.conv1_act = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=self.kernel, stride=self.stride, padding=self.padding)
        self.conv2_batch = nn.BatchNorm2d(124)
        self.conv2_act = nn.ReLU()
        # self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=self.kernel, stride=self.stride, padding=self.padding)
        # self.conv3_act = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=self.pool)

        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=self.kernel, stride=self.stride, padding=self.padding)
        self.conv4_batch = nn.BatchNorm2d(60)
        self.conv4_act = nn.ReLU()
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=self.kernel, stride=self.stride, padding=self.padding)
        self.conv5_batch = nn.BatchNorm2d(58)
        self.conv5_act = nn.ReLU()
        # self.conv6 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=self.kernel, stride=self.stride, padding=self.padding)
        # self.conv6_act = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=self.pool)

        self.conv7 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=self.kernel, stride=self.stride, padding=self.padding)
        self.conv7_batch = nn.BatchNorm2d(27)
        self.conv7_act = nn.ReLU()
        self.conv8 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=self.kernel, stride=self.stride, padding=self.padding)
        self.conv8_batch = nn.BatchNorm2d(25)
        self.conv8_act = nn.ReLU()
        # self.conv9 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=self.kernel, stride=self.stride, padding=self.padding)
        # self.conv9_act = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=self.pool)

        self.flatten = nn.Flatten()
        self.flatten_drop = nn.Dropout(0.5)

        self.fc1 = nn.Linear(in_features=128*12*12, out_features=256)
        self.fc1_batch = nn.BatchNorm1d(256)
        self.fc1_drop = nn.Dropout(0.5)
        self.fc1_act = nn.ReLU()
        self.fc2 = nn.Linear(in_features=256, out_features=64)
        self.fc2_batch = nn.BatchNorm1d(64)
        self.fc2_drop = nn.Dropout(0.5)
        self.fc2_act = nn.ReLU()
        self.fc3 = nn.Linear(in_features=64, out_features=self.n_classes)
        self.fc3_act = nn.Softmax()

    def forward(self, x):
        x = self.conv1_act(self.conv1(x))
        x = self.conv2_act(self.conv2(x))
        x = self.pool1(x)

        x = self.conv4_act(self.conv4(x))
        x = self.conv5_act(self.conv5(x))
        x = self.pool2(x)

        x = self.conv7_act(self.conv7(x))
        x = self.conv8_act(self.conv8(x))
        x = self.pool3(x)

        x = self.flatten_drop(self.flatten(x))

        x = self.fc1_act(self.fc1_drop(self.fc1_batch(self.fc1(x))))
        x = self.fc2_act(self.fc2_drop(self.fc2_batch(self.fc2(x))))
        x = self.fc3_act(self.fc3(x))

        return x


class CustomNetSmall(nn.Module):

    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes

        self.kernel = 3
        self.stride = 1
        self.padding = 0

        self.pool = 2

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=self.kernel, stride=self.stride, padding=self.padding)
        self.conv1_batch = nn.BatchNorm2d(126)
        self.conv1_act = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=self.kernel, stride=self.stride, padding=self.padding)
        self.conv2_batch = nn.BatchNorm2d(124)
        self.conv2_act = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=self.pool)

        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=self.kernel, stride=self.stride, padding=self.padding)
        self.conv4_batch = nn.BatchNorm2d(60)
        self.conv4_act = nn.ReLU()
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=self.kernel, stride=self.stride, padding=self.padding)
        self.conv5_batch = nn.BatchNorm2d(58)
        self.conv5_act = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=self.pool)

        self.flatten = nn.Flatten()
        self.flatten_drop = nn.Dropout(0.5)

        self.fc1 = nn.Linear(in_features=64*29*29, out_features=1024)
        self.fc1_batch = nn.BatchNorm1d(1024)
        self.fc1_drop = nn.Dropout(0.5)
        self.fc1_act = nn.ReLU()
        self.fc2 = nn.Linear(in_features=1024, out_features=256)
        self.fc2_batch = nn.BatchNorm1d(256)
        self.fc2_drop = nn.Dropout(0.5)
        self.fc2_act = nn.ReLU()
        self.fc3 = nn.Linear(in_features=256, out_features=64)
        self.fc3_batch = nn.BatchNorm1d(64)
        self.fc3_drop = nn.Dropout(0.5)
        self.fc3_act = nn.ReLU()
        self.fc4 = nn.Linear(in_features=64, out_features=self.n_classes)
        self.fc4_act = nn.Softmax()

    def forward(self, x):
        x = self.conv1_act(self.conv1(x))
        x = self.conv2_act(self.conv2(x))
        x = self.pool1(x)

        x = self.conv4_act(self.conv4(x))
        x = self.conv5_act(self.conv5(x))
        x = self.pool2(x)

        x = self.flatten_drop(self.flatten(x))

        x = self.fc1_act(self.fc1_drop(self.fc1_batch(self.fc1(x))))
        x = self.fc2_act(self.fc2_drop(self.fc2_batch(self.fc2(x))))
        x = self.fc3_act(self.fc3_drop(self.fc3_batch(self.fc3(x))))
        x = self.fc4_act(self.fc4(x))

        return x


class CustomNetSmallLogits(nn.Module):

    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes

        self.kernel = 3
        self.stride = 1
        self.padding = 0

        self.pool = 2

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=self.kernel, stride=self.stride,
                               padding=self.padding)
        self.conv1_batch = nn.BatchNorm2d(126)
        self.conv1_act = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=self.kernel, stride=self.stride,
                               padding=self.padding)
        self.conv2_batch = nn.BatchNorm2d(124)
        self.conv2_act = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=self.pool)

        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=self.kernel, stride=self.stride,
                               padding=self.padding)
        self.conv4_batch = nn.BatchNorm2d(60)
        self.conv4_act = nn.ReLU()
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=self.kernel, stride=self.stride,
                               padding=self.padding)
        self.conv5_batch = nn.BatchNorm2d(58)
        self.conv5_act = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=self.pool)

        self.flatten = nn.Flatten()
        self.flatten_drop = nn.Dropout(0.5)

        self.fc1 = nn.Linear(in_features=64 * 29 * 29, out_features=1024)
        self.fc1_batch = nn.BatchNorm1d(1024)
        self.fc1_drop = nn.Dropout(0.5)
        self.fc1_act = nn.ReLU()
        self.fc2 = nn.Linear(in_features=1024, out_features=256)
        self.fc2_batch = nn.BatchNorm1d(256)
        self.fc2_drop = nn.Dropout(0.5)
        self.fc2_act = nn.ReLU()
        self.fc3 = nn.Linear(in_features=256, out_features=64)
        self.fc3_batch = nn.BatchNorm1d(64)
        self.fc3_drop = nn.Dropout(0.5)
        self.fc3_act = nn.ReLU()
        self.fc4 = nn.Linear(in_features=64, out_features=self.n_classes)

    def forward(self, x):
        x = self.conv1_act(self.conv1(x))
        x = self.conv2_act(self.conv2(x))
        x = self.pool1(x)

        x = self.conv4_act(self.conv4(x))
        x = self.conv5_act(self.conv5(x))
        x = self.pool2(x) # GAP instead results in size [N, 64]

        x = self.flatten_drop(self.flatten(x))

        x = self.fc1_act(self.fc1_drop(self.fc1_batch(self.fc1(x))))
        x = self.fc2_act(self.fc2_drop(self.fc2_batch(self.fc2(x))))
        x = self.fc3_act(self.fc3_drop(self.fc3_batch(self.fc3(x))))
        x = self.fc4(x)

        return x


class CustomNetSmallGAPLogits(nn.Module):

    def __init__(self, n_classes, in_channels=1):
        super().__init__()
        self.n_classes = n_classes
        self.in_channels = in_channels

        self.kernel = 3
        self.stride = 1
        self.padding = 0

        self.pool = 2

        self.conv1 = nn.Conv2d(in_channels=self.in_channels, out_channels=32, kernel_size=self.kernel, stride=self.stride,
                               padding=self.padding)
        self.conv1_batch = nn.BatchNorm2d(126)
        self.conv1_act = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=self.kernel, stride=self.stride,
                               padding=self.padding)
        self.conv2_batch = nn.BatchNorm2d(124)
        self.conv2_act = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=self.pool)

        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=self.kernel, stride=self.stride,
                               padding=self.padding)
        self.conv4_batch = nn.BatchNorm2d(60)
        self.conv4_act = nn.ReLU()
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=self.kernel, stride=self.stride,
                               padding=self.padding)
        self.conv5_batch = nn.BatchNorm2d(58)
        self.conv5_act = nn.ReLU()

        self.classifier = nn.Linear(in_features=64, out_features=self.n_classes)

    def forward(self, x):
        x = self.conv1_act(self.conv1(x))
        x = self.conv2_act(self.conv2(x))
        x = self.pool1(x)

        x = self.conv4_act(self.conv4(x))
        x = self.conv5_act(self.conv5(x))

        # GAP layer
        x = torch.mean(x.view(x.size(0), x.size(1), -1), dim=2)

        logits = self.classifier(x)
        return logits, x


class CustomNetBigGAPLogits(nn.Module):

    def __init__(self, n_classes, in_channels=1):
        super().__init__()
        self.n_classes = n_classes
        self.in_channels = in_channels

        self.kernel = 3
        self.stride = 1
        self.padding = 0

        self.pool = 2

        self.conv1 = nn.Conv2d(in_channels=self.in_channels, out_channels=512, kernel_size=self.kernel, stride=self.stride,
                               padding=self.padding)
        self.conv1_batch = nn.BatchNorm2d(126)
        self.conv1_act = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=512, out_channels=2048, kernel_size=self.kernel, stride=self.stride,
                               padding=self.padding)
        self.conv2_batch = nn.BatchNorm2d(124)
        self.conv2_act = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=self.pool)

        self.conv4 = nn.Conv2d(in_channels=2048, out_channels=512, kernel_size=self.kernel, stride=self.stride,
                               padding=self.padding)
        self.conv4_batch = nn.BatchNorm2d(60)
        self.conv4_act = nn.ReLU()
        self.conv5 = nn.Conv2d(in_channels=512, out_channels=64, kernel_size=self.kernel, stride=self.stride,
                               padding=self.padding)
        self.conv5_batch = nn.BatchNorm2d(58)
        self.conv5_act = nn.ReLU()

        self.classifier = nn.Linear(in_features=64, out_features=self.n_classes)

    def forward(self, x):
        x = self.conv1_act(self.conv1(x))
        x = self.conv2_act(self.conv2(x))
        x = self.pool1(x)

        x = self.conv4_act(self.conv4(x))
        x = self.conv5_act(self.conv5(x))

        # GAP layer
        x = torch.mean(x.view(x.size(0), x.size(1), -1), dim=2)

        logits = self.classifier(x)
        return logits, x


class CustomNetSmallNoGAPLogits(CustomNetSmallGAPLogits):

    def __init__(self, n_classes, in_channels=1):
        super().__init__(n_classes, in_channels)

        self.flatten = nn.Flatten()
        self.flatten_drop = nn.Dropout(0.5)

        self.fc1 = nn.Linear(in_features=64 * 58 * 58, out_features=1024)
        self.fc1_batch = nn.BatchNorm1d(1024)
        self.fc1_drop = nn.Dropout(0.5)
        self.fc1_act = nn.ReLU()
        self.fc2 = nn.Linear(in_features=1024, out_features=256)
        self.fc2_batch = nn.BatchNorm1d(256)
        self.fc2_drop = nn.Dropout(0.5)
        self.fc2_act = nn.ReLU()
        self.fc3 = nn.Linear(in_features=256, out_features=64)
        self.fc3_batch = nn.BatchNorm1d(64)
        self.fc3_act = nn.ReLU()


    def forward(self, x):
        x = self.conv1_act(self.conv1(x))
        x = self.conv2_act(self.conv2(x))
        x = self.pool1(x)

        x = self.conv4_act(self.conv4(x))
        x = self.conv5_act(self.conv5(x))

        x = self.flatten_drop(self.flatten(x))

        x = self.fc1_act(self.fc1_drop(self.fc1_batch(self.fc1(x))))
        x = self.fc2_act(self.fc2_drop(self.fc2_batch(self.fc2(x))))
        x = self.fc3_act(self.fc3_batch(self.fc3(x)))

        logits = self.classifier(x)
        return logits, x


class SoftmaxWrapper(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        output = self.model(x)

        return self.softmax(output)


class SoftmaxLogitWrapper(SoftmaxWrapper):

    def __init__(self, model):
        super().__init__(model)

    def forward(self, x):
        logits, x = self.model(x)

        return self.softmax(logits)


class CustomNetVerySmall(nn.Module):

    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes

        self.kernel = 7
        self.stride = 1
        self.padding = 0

        self.pool = 7

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=self.kernel, stride=self.stride, padding=self.padding)
        self.conv1_batch = nn.BatchNorm2d(122)
        self.conv1_act = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=self.kernel, stride=self.stride, padding=self.padding)
        self.conv2_batch = nn.BatchNorm2d(116)
        self.conv2_act = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=self.pool)

        self.flatten = nn.Flatten()
        self.flatten_drop = nn.Dropout(0.5)

        self.fc1 = nn.Linear(in_features=32*16*16, out_features=128)
        self.fc1_batch = nn.BatchNorm1d(128)
        self.fc1_drop = nn.Dropout(0.5)
        self.fc1_act = nn.ReLU()
        self.fc2 = nn.Linear(in_features=128, out_features=self.n_classes)
        # self.fc2_act = nn.Softmax()

    def forward(self, x):
        x = self.conv1_act(self.conv1(x))
        x = self.conv2_act(self.conv2(x))
        x = self.pool1(x)

        x = self.flatten_drop(self.flatten(x))

        x = self.fc1_act(self.fc1_drop(self.fc1_batch(self.fc1(x))))
        x = self.fc2(x)
        # x = self.fc2_act()

        return x


class CustomNetSmallBinary(nn.Module):

    def __init__(self, in_channels=1):
        super().__init__()
        self.n_classes = 1

        self.kernel = 3
        self.stride = 1
        self.padding = 0

        self.pool = 2

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=self.kernel, stride=self.stride, padding=self.padding)
        self.conv1_batch = nn.BatchNorm2d(126)
        self.conv1_act = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=self.kernel, stride=self.stride, padding=self.padding)
        self.conv2_batch = nn.BatchNorm2d(124)
        self.conv2_act = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=self.pool)

        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=self.kernel, stride=self.stride, padding=self.padding)
        self.conv4_batch = nn.BatchNorm2d(60)
        self.conv4_act = nn.ReLU()
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=self.kernel, stride=self.stride, padding=self.padding)
        self.conv5_batch = nn.BatchNorm2d(58)
        self.conv5_act = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=self.pool)

        self.flatten = nn.Flatten()
        self.flatten_drop = nn.Dropout(0.5)

        self.fc1 = nn.Linear(in_features=64*29*29, out_features=1024)
        self.fc1_batch = nn.BatchNorm1d(1024)
        self.fc1_drop = nn.Dropout(0.5)
        self.fc1_act = nn.ReLU()
        self.fc2 = nn.Linear(in_features=1024, out_features=256)
        self.fc2_batch = nn.BatchNorm1d(256)
        self.fc2_drop = nn.Dropout(0.5)
        self.fc2_act = nn.ReLU()
        self.fc3 = nn.Linear(in_features=256, out_features=64)
        self.fc3_batch = nn.BatchNorm1d(64)
        self.fc3_drop = nn.Dropout(0.5)
        self.fc3_act = nn.ReLU()
        self.fc4 = nn.Linear(in_features=64, out_features=self.n_classes)
        self.fc4_act = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1_act(self.conv1(x))
        x = self.conv2_act(self.conv2(x))
        x = self.pool1(x)

        x = self.conv4_act(self.conv4(x))
        x = self.conv5_act(self.conv5(x))
        x = self.pool2(x)

        x = self.flatten_drop(self.flatten(x))

        x = self.fc1_act(self.fc1_drop(self.fc1_batch(self.fc1(x))))
        x = self.fc2_act(self.fc2_drop(self.fc2_batch(self.fc2(x))))
        x = self.fc3_act(self.fc3_drop(self.fc3_batch(self.fc3(x))))
        x = self.fc4(x)
        x = self.fc4_act(x)

        return x


class CustomNetBinary(nn.Module):

    def __init__(self, in_channels=1):
        super().__init__()
        self.n_classes = 1

        self.kernel = 3
        self.stride = 1
        self.padding = 0

        self.pool = 2

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=self.kernel, stride=self.stride,
                               padding=self.padding)
        self.conv1_batch = nn.BatchNorm2d(126)
        self.conv1_act = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=self.kernel, stride=self.stride,
                               padding=self.padding)
        self.conv2_batch = nn.BatchNorm2d(124)
        self.conv2_act = nn.ReLU()
        # self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=self.kernel, stride=self.stride, padding=self.padding)
        # self.conv3_act = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=self.pool)

        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=self.kernel, stride=self.stride,
                               padding=self.padding)
        self.conv4_batch = nn.BatchNorm2d(60)
        self.conv4_act = nn.ReLU()
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=self.kernel, stride=self.stride,
                               padding=self.padding)
        self.conv5_batch = nn.BatchNorm2d(58)
        self.conv5_act = nn.ReLU()
        # self.conv6 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=self.kernel, stride=self.stride, padding=self.padding)
        # self.conv6_act = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=self.pool)

        self.conv7 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=self.kernel, stride=self.stride,
                               padding=self.padding)
        self.conv7_batch = nn.BatchNorm2d(27)
        self.conv7_act = nn.ReLU()
        self.conv8 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=self.kernel, stride=self.stride,
                               padding=self.padding)
        self.conv8_batch = nn.BatchNorm2d(25)
        self.conv8_act = nn.ReLU()
        # self.conv9 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=self.kernel, stride=self.stride, padding=self.padding)
        # self.conv9_act = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=self.pool)

        self.flatten = nn.Flatten()
        self.flatten_drop = nn.Dropout(0.5)

        self.fc1 = nn.Linear(in_features=128 * 12 * 12, out_features=256)
        self.fc1_batch = nn.BatchNorm1d(256)
        self.fc1_drop = nn.Dropout(0.5)
        self.fc1_act = nn.ReLU()
        self.fc2 = nn.Linear(in_features=256, out_features=64)
        self.fc2_batch = nn.BatchNorm1d(64)
        self.fc2_drop = nn.Dropout(0.5)
        self.fc2_act = nn.ReLU()
        self.fc3 = nn.Linear(in_features=64, out_features=self.n_classes)
        self.fc3_act = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1_act(self.conv1(x))
        x = self.conv2_act(self.conv2(x))
        x = self.pool1(x)

        x = self.conv4_act(self.conv4(x))
        x = self.conv5_act(self.conv5(x))
        x = self.pool2(x)

        x = self.conv7_act(self.conv7(x))
        x = self.conv8_act(self.conv8(x))
        x = self.pool3(x)

        x = self.flatten_drop(self.flatten(x))

        x = self.fc1_act(self.fc1_drop(self.fc1_batch(self.fc1(x))))
        x = self.fc2_act(self.fc2_drop(self.fc2_batch(self.fc2(x))))
        x = self.fc3_act(self.fc3(x))

        return x


class CustomNetExtremelySmallBinary(nn.Module):

    def __init__(self, in_channels=1):
        super().__init__()
        self.n_classes = 1

        self.kernel = 7
        self.stride = 1
        self.padding = 0

        self.pool = 7

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=self.kernel, stride=self.stride, padding=self.padding)
        self.conv1_batch = nn.BatchNorm2d(122)
        self.conv1_act = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=self.pool)

        self.flatten = nn.Flatten()
        self.flatten_drop = nn.Dropout(0.5)

        self.fc1 = nn.Linear(in_features=32*17*17, out_features=self.n_classes)
        self.fc1_act = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1_act(self.conv1(x))
        x = self.pool1(x)

        x = self.flatten_drop(self.flatten(x))

        x = self.fc1(x)
        x = self.fc1_act(x)

        return x


class ModifiedAlexNet(CustomNetSmallGAPLogits):  # inheritance to be used with SoftMaxWrapper

    def __init__(self, n_classes, in_channels=1):
        super().__init__(n_classes, in_channels)

        self.c1 = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=64,
            kernel_size=11,
            stride=4,
            padding=2,
        )
        self.c1_act = nn.ReLU()
        self.p1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.batch1 = nn.BatchNorm2d(64)

        self.c2 = nn.Conv2d(
            in_channels=64,
            out_channels=192,
            kernel_size=5,
            padding=2,
        )
        self.c2_act = nn.ReLU()
        self.p2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.batch2 = nn.BatchNorm2d(192)

        self.c3 = nn.Conv2d(
            in_channels=192,
            out_channels=384,
            kernel_size=3,
            padding=1
        )
        self.c3_act = nn.ReLU()
        self.batch3 = nn.BatchNorm2d(384)

        self.c4 = nn.Conv2d(
            in_channels=384,
            out_channels=256,
            kernel_size=3,
            padding=1
        )
        self.c4_act = nn.ReLU()
        self.batch4 = nn.BatchNorm2d(256)

        self.c5 = nn.Conv2d(
            in_channels=256,
            out_channels=256,
            kernel_size=3,
            padding=1
        )
        self.c5_act = nn.ReLU()
        self.p5 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.batch5 = nn.BatchNorm2d(256)

        self.flatten = nn.Flatten()

        self.d1 = nn.Dropout()

        self.f1 = nn.Linear(256*3*3, 4096)
        self.f1_act = nn.ReLU()
        self.f1_d = nn.Dropout()
        self.f1_batch = nn.BatchNorm1d(4096)

        self.f2 = nn.Linear(4096, 4096)
        self.f2_act = nn.ReLU()
        self.f2_d = nn.Dropout()
        self.f2_batch = nn.BatchNorm1d(4096)

        self.f3 = nn.Linear(4096, 64)
        self.f3_act = nn.ReLU()
        self.f3_d = nn.Dropout()
        self.f3_batch = nn.BatchNorm1d(64)

        self.classifier = nn.Linear(in_features=64, out_features=self.n_classes)

    def forward(self, x):
        x = self.c1(x)
        x = self.c1_act(x)
        x = self.p1(x)
        x = self.batch1(x)

        x = self.c2(x)
        x = self.c2_act(x)
        x = self.p2(x)
        x = self.batch2(x)

        x = self.c3(x)
        x = self.c3_act(x)
        x = self.batch3(x)

        x = self.c4(x)
        x = self.c4_act(x)
        x = self.batch4(x)

        x = self.c5(x)
        x = self.c5_act(x)
        x = self.p5(x)
        x = self.batch5(x)

        x = self.flatten(x)
        x = self.d1(x)

        x = self.f1(x)
        x = self.f1_act(x)
        x = self.f1_d(x)
        x = self.f1_batch(x)

        x = self.f2(x)
        x = self.f2_act(x)
        x = self.f2_d(x)
        x = self.f2_batch(x)

        x = self.f3(x)
        x = self.f3_act(x)
        x = self.f3_d(x)
        x = self.f3_batch(x)

        logits = self.classifier(x)
        return logits, x
