import torch
from torch import nn

# paper: Isola


class DPatchGanSmallWithClass(nn.Module):

    def __init__(self, in_channels=2, input_size=128):
        super().__init__()
        self.in_channels = in_channels
        self.input_size = input_size

        self.kernel = 4
        self.stride = 2
        self.padding = 1

        self.leaky_slope = 0.2

        self.emb_size = 8
        self.emb_layer = nn.Embedding(2, self.emb_size * self.emb_size)
        self.upsample_layer = nn.Upsample(size=(self.input_size, self.input_size))

        self.l1 = nn.Sequential(
            nn.Conv2d(self.in_channels, 64, kernel_size=self.kernel, stride=self.stride, padding=self.padding),
            nn.LeakyReLU(negative_slope=self.leaky_slope)
        )

        self.l2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=self.kernel, stride=self.stride, padding=self.padding),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=self.leaky_slope)
        )

        self.l3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=self.kernel, stride=1, padding=self.padding),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=self.leaky_slope)
        )

        self.l4 = nn.Sequential(
            nn.Conv2d(256, 1, kernel_size=self.kernel, stride=1, padding=self.padding),
            nn.Sigmoid()
        )

    def forward(self, x, labels):
        self.assert_size(x, self.in_channels - 1, self.input_size)
        labels_embedded = self.emb_layer(labels)  # size (n, 10)
        labels_embedded = labels_embedded.reshape(labels_embedded.size(0), 1, self.emb_size, self.emb_size)
        labels_upsampled = self.upsample_layer(labels_embedded)

        x_cat = torch.cat([x, labels_upsampled], dim=1)


        self.assert_size(x_cat, self.in_channels, self.input_size)

        l1_out = self.l1(x_cat)
        l2_out = self.l2(l1_out)
        l3_out = self.l3(l2_out)
        l4_out = self.l4(l3_out)

        self.assert_size(l4_out, 1, 30)
        return l4_out

    def assert_size(self, tensor, channel, size):
        assert tensor.shape[1:] == (channel, size, size)


class DPatchGanWithClass(nn.Module):

    def __init__(self, in_channels=2, input_size=128):
        super().__init__()
        self.in_channels = in_channels
        self.input_size = input_size

        self.kernel = 4
        self.stride = 2
        self.padding = 1

        self.leaky_slope = 0.2

        self.emb_size = 8
        self.emb_layer = nn.Embedding(2, self.emb_size * self.emb_size)
        self.upsample_layer = nn.Upsample(size=(self.input_size, self.input_size))

        self.l1 = nn.Sequential(
            nn.Conv2d(self.in_channels, 256, kernel_size=self.kernel, stride=self.stride, padding=self.padding),
            nn.LeakyReLU(negative_slope=self.leaky_slope)
        )

        self.l2 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=self.kernel, stride=self.stride, padding=self.padding),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=self.leaky_slope)
        )

        self.l3 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=self.kernel, stride=1, padding=self.padding),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=self.leaky_slope)
        )

        self.l4 = nn.Sequential(
            nn.Conv2d(256, 1, kernel_size=self.kernel, stride=1, padding=self.padding),
            nn.Sigmoid()
        )

    def forward(self, x, labels):
        self.assert_size(x, self.in_channels - 1, self.input_size)
        labels_embedded = self.emb_layer(labels)  # size (n, 10)
        labels_embedded = labels_embedded.reshape(labels_embedded.size(0), 1, self.emb_size, self.emb_size)
        labels_upsampled = self.upsample_layer(labels_embedded)

        x_cat = torch.cat([x, labels_upsampled], dim=1)


        self.assert_size(x_cat, self.in_channels, self.input_size)

        l1_out = self.l1(x_cat)
        l2_out = self.l2(l1_out)
        l3_out = self.l3(l2_out)
        l4_out = self.l4(l3_out)

        self.assert_size(l4_out, 1, 30)
        return l4_out

    def assert_size(self, tensor, channel, size):
        assert tensor.shape[1:] == (channel, size, size)


class DPatchGanBigWithClass(nn.Module):

    def __init__(self, in_channels=2, input_size=128):
        super().__init__()
        self.in_channels = in_channels
        self.input_size = input_size

        self.kernel = 4
        self.stride = 2
        self.padding = 1

        self.leaky_slope = 0.2

        self.emb_size = 8
        self.emb_layer = nn.Embedding(2, self.emb_size * self.emb_size)
        self.upsample_layer = nn.Upsample(size=(self.input_size, self.input_size))

        self.l1 = nn.Sequential(
            nn.Conv2d(self.in_channels, 512, kernel_size=self.kernel, stride=self.stride, padding=self.padding),
            nn.LeakyReLU(negative_slope=self.leaky_slope)
        )

        self.l2 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=self.kernel, stride=self.stride, padding=self.padding),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(negative_slope=self.leaky_slope)
        )

        self.l3 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=self.kernel, stride=1, padding=self.padding),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=self.leaky_slope)
        )

        self.l4 = nn.Sequential(
            nn.Conv2d(512, 1, kernel_size=self.kernel, stride=1, padding=self.padding),
            nn.Sigmoid()
        )

    def forward(self, x, labels):
        self.assert_size(x, self.in_channels - 1, self.input_size)
        labels_embedded = self.emb_layer(labels)  # size (n, 10)
        labels_embedded = labels_embedded.reshape(labels_embedded.size(0), 1, self.emb_size, self.emb_size)
        labels_upsampled = self.upsample_layer(labels_embedded)

        x_cat = torch.cat([x, labels_upsampled], dim=1)


        self.assert_size(x_cat, self.in_channels, self.input_size)

        l1_out = self.l1(x_cat)
        l2_out = self.l2(l1_out)
        l3_out = self.l3(l2_out)
        l4_out = self.l4(l3_out)

        self.assert_size(l4_out, 1, 30)
        return l4_out

    def assert_size(self, tensor, channel, size):
        assert tensor.shape[1:] == (channel, size, size)


class DPatchGanHugeWithClass(nn.Module):

    def __init__(self, in_channels=2, input_size=128):
        super().__init__()
        self.in_channels = in_channels
        self.input_size = input_size

        self.kernel = 4
        self.stride = 2
        self.padding = 1

        self.leaky_slope = 0.2

        self.emb_size = 8
        self.emb_layer = nn.Embedding(2, self.emb_size * self.emb_size)
        self.upsample_layer = nn.Upsample(size=(self.input_size, self.input_size))

        self.l1 = nn.Sequential(
            nn.Conv2d(self.in_channels, 1024, kernel_size=self.kernel, stride=self.stride, padding=self.padding),
            nn.LeakyReLU(negative_slope=self.leaky_slope)
        )

        self.l2 = nn.Sequential(
            nn.Conv2d(1024, 2048, kernel_size=self.kernel, stride=self.stride, padding=self.padding),
            nn.BatchNorm2d(2048),
            nn.LeakyReLU(negative_slope=self.leaky_slope)
        )

        self.l3 = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=self.kernel, stride=1, padding=self.padding),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(negative_slope=self.leaky_slope)
        )

        self.l4 = nn.Sequential(
            nn.Conv2d(1024, 1, kernel_size=self.kernel, stride=1, padding=self.padding),
            nn.Sigmoid()
        )

    def forward(self, x, labels):
        self.assert_size(x, self.in_channels - 1, self.input_size)
        labels_embedded = self.emb_layer(labels)  # size (n, 10)
        labels_embedded = labels_embedded.reshape(labels_embedded.size(0), 1, self.emb_size, self.emb_size)
        labels_upsampled = self.upsample_layer(labels_embedded)

        x_cat = torch.cat([x, labels_upsampled], dim=1)


        self.assert_size(x_cat, self.in_channels, self.input_size)

        l1_out = self.l1(x_cat)
        l2_out = self.l2(l1_out)
        l3_out = self.l3(l2_out)
        l4_out = self.l4(l3_out)

        self.assert_size(l4_out, 1, 30)
        return l4_out

    def assert_size(self, tensor, channel, size):
        assert tensor.shape[1:] == (channel, size, size)


class DPatchGanHugerWithClass(nn.Module):

    def __init__(self, in_channels=2, input_size=128):
        super().__init__()
        self.in_channels = in_channels
        self.input_size = input_size

        self.kernel = 4
        self.stride = 2
        self.padding = 1

        self.leaky_slope = 0.2

        self.emb_size = 8
        self.emb_layer = nn.Embedding(2, self.emb_size * self.emb_size)
        self.upsample_layer = nn.Upsample(size=(self.input_size, self.input_size))

        self.l1 = nn.Sequential(
            nn.Conv2d(self.in_channels, 1600, kernel_size=self.kernel, stride=self.stride, padding=self.padding),
            nn.LeakyReLU(negative_slope=self.leaky_slope)
        )

        self.l2 = nn.Sequential(
            nn.Conv2d(1600, 3000, kernel_size=self.kernel, stride=self.stride, padding=self.padding),
            nn.BatchNorm2d(3000),
            nn.LeakyReLU(negative_slope=self.leaky_slope)
        )

        self.l3 = nn.Sequential(
            nn.Conv2d(3000, 1600, kernel_size=self.kernel, stride=1, padding=self.padding),
            nn.BatchNorm2d(1600),
            nn.LeakyReLU(negative_slope=self.leaky_slope)
        )

        self.l4 = nn.Sequential(
            nn.Conv2d(1600, 1, kernel_size=self.kernel, stride=1, padding=self.padding),
            nn.Sigmoid()
        )

    def forward(self, x, labels):
        self.assert_size(x, self.in_channels - 1, self.input_size)
        labels_embedded = self.emb_layer(labels)
        labels_embedded = labels_embedded.reshape(labels_embedded.size(0), 1, self.emb_size, self.emb_size)
        labels_upsampled = self.upsample_layer(labels_embedded)

        x_cat = torch.cat([x, labels_upsampled], dim=1)


        self.assert_size(x_cat, self.in_channels, self.input_size)

        l1_out = self.l1(x_cat)
        l2_out = self.l2(l1_out)
        l3_out = self.l3(l2_out)
        l4_out = self.l4(l3_out)

        self.assert_size(l4_out, 1, 30)
        return l4_out

    def assert_size(self, tensor, channel, size):
        assert tensor.shape[1:] == (channel, size, size)
