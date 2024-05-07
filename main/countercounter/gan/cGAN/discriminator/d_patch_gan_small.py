from torch import nn

# paper: Isola


class DPatchGanSmall(nn.Module):

    def __init__(self, in_channels=1, input_size=128):
        super().__init__()
        self.in_channels = in_channels
        self.input_size = input_size

        self.kernel = 4
        self.stride = 2
        self.padding = 1

        self.leaky_slope = 0.2

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

    def forward(self, x):
        self.assert_size(x, self.in_channels, self.input_size)

        l1_out = self.l1(x)
        l2_out = self.l2(l1_out)
        l3_out = self.l3(l2_out)
        l4_out = self.l4(l3_out)

        self.assert_size(l4_out, 1, 30)
        return l4_out

    def assert_size(self, tensor, channel, size):
        assert tensor.shape[1:] == (channel, size, size)


class DPatchGanBig(nn.Module):

    def __init__(self, in_channels=1, input_size=128):
        super().__init__()
        self.in_channels = in_channels
        self.input_size = input_size

        self.kernel = 4
        self.stride = 2
        self.padding = 1

        self.leaky_slope = 0.2

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

    def forward(self, x):
        self.assert_size(x, self.in_channels, self.input_size)

        l1_out = self.l1(x)
        l2_out = self.l2(l1_out)
        l3_out = self.l3(l2_out)
        l4_out = self.l4(l3_out)

        self.assert_size(l4_out, 1, 30)
        return l4_out

    def assert_size(self, tensor, channel, size):
        assert tensor.shape[1:] == (channel, size, size)


class DPatchGanHuge(nn.Module):

    def __init__(self, in_channels=1, input_size=128):
        super().__init__()
        self.in_channels = in_channels
        self.input_size = input_size

        self.kernel = 4
        self.stride = 2
        self.padding = 1

        self.leaky_slope = 0.2

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

    def forward(self, x):
        self.assert_size(x, self.in_channels, self.input_size)

        l1_out = self.l1(x)
        l2_out = self.l2(l1_out)
        l3_out = self.l3(l2_out)
        l4_out = self.l4(l3_out)

        self.assert_size(l4_out, 1, 30)
        return l4_out

    def assert_size(self, tensor, channel, size):
        assert tensor.shape[1:] == (channel, size, size)


class DPatchGanHuger(nn.Module):

    def __init__(self, in_channels=1, input_size=128):
        super().__init__()
        self.in_channels = in_channels
        self.input_size = input_size

        self.kernel = 4
        self.stride = 2
        self.padding = 1

        self.leaky_slope = 0.2

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

    def forward(self, x):
        self.assert_size(x, self.in_channels, self.input_size)

        l1_out = self.l1(x)
        l2_out = self.l2(l1_out)
        l3_out = self.l3(l2_out)
        l4_out = self.l4(l3_out)

        self.assert_size(l4_out, 1, 30)
        return l4_out

    def assert_size(self, tensor, channel, size):
        assert tensor.shape[1:] == (channel, size, size)