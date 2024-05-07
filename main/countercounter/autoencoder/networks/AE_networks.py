import torch.nn as nn


class AEV2Net(nn.Module):

    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.kernel = 3
        self.padding = 1
        self.stride = 2
        self.output_padding = 1

        self.l1 = nn.Conv2d(self.in_channels, 32, kernel_size=self.kernel, padding=self.padding, stride=self.stride)
        self._l1_norm = nn.BatchNorm2d(32)
        self._l1_act = nn.ReLU()
        self.l2 = nn.Conv2d(32, 64, kernel_size=self.kernel, padding=self.padding, stride=self.stride)
        self._l2_norm = nn.BatchNorm2d(64)
        self._l2_act = nn.ReLU()
        self.l3 = nn.Conv2d(64, 128, kernel_size=self.kernel, padding=self.padding, stride=self.stride)
        self._l3_norm = nn.BatchNorm2d(128)
        self._l3_act = nn.ReLU()
        self.l4 = nn.Conv2d(128, 128, kernel_size=self.kernel, padding=self.padding, stride=self.stride)
        self._l4_norm = nn.BatchNorm2d(128)
        self._l4_act = nn.ReLU()

        self.l5 = nn.ConvTranspose2d(128, 128, kernel_size=self.kernel, padding=self.padding, stride=self.stride, output_padding=self.output_padding)
        self._l5_norm = nn.BatchNorm2d(128)
        self._l5_act = nn.ReLU()
        self.l6 = nn.ConvTranspose2d(128, 64, kernel_size=self.kernel, padding=self.padding, stride=self.stride, output_padding=self.output_padding)
        self._l6_norm = nn.BatchNorm2d(64)
        self._l6_act = nn.ReLU()
        self.l7 = nn.ConvTranspose2d(64, 32, kernel_size=self.kernel, padding=self.padding, stride=self.stride, output_padding=self.output_padding)
        self._l7_norm = nn.BatchNorm2d(32)
        self._l7_act = nn.ReLU()
        self.l8 = nn.ConvTranspose2d(32, self.in_channels, kernel_size=self.kernel, padding=self.padding, stride=self.stride, output_padding=self.output_padding)
        self._l8_act = nn.Tanh()

    def forward(self, x):
        assert x.shape[1:] == (self.in_channels, 128, 128)

        x_1 = self._l1_act(self._l1_norm(self.l1(x)))
        x_2 = self._l2_act(self._l2_norm(self.l2(x_1)))
        x_3 = self._l3_act(self._l3_norm(self.l3(x_2)))
        x_4 = self._l4_act(self._l4_norm(self.l4(x_3)))
        x_5 = self._l5_act(self._l5_norm(self.l5(x_4)))
        x_6 = self._l6_act(self._l6_norm(self.l6(x_5)))
        x_7 = self._l7_act(self._l7_norm(self.l7(x_6)))
        x_8 = self._l8_act(self.l8(x_7))

        assert x_8.shape[1:] == (self.in_channels, 128, 128)
        return x_8


class AEV2BiggerNet(nn.Module):

    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.kernel = 3
        self.padding = 1
        self.stride = 2
        self.output_padding = 1

        self.scale_factor = 2
        self.upsample_mode = 'nearest'
        self.padding_reflection_pad = 1
        self.padding_conv2d_replacement = 0
        self.kernel_conv2d_replacement = 3
        self.stride_conv2d_replacement = 1

        self.l1 = nn.Conv2d(self.in_channels, 32, kernel_size=self.kernel, padding=self.padding, stride=self.stride)
        self._l1_norm = nn.BatchNorm2d(32)
        self._l1_act = nn.ReLU()
        self.l2 = nn.Conv2d(32, 64, kernel_size=self.kernel, padding=self.padding, stride=self.stride)
        self._l2_norm = nn.BatchNorm2d(64)
        self._l2_act = nn.ReLU()
        self.l3 = nn.Conv2d(64, 128, kernel_size=self.kernel, padding=self.padding, stride=self.stride)
        self._l3_norm = nn.BatchNorm2d(128)
        self._l3_act = nn.ReLU()
        self.l4 = nn.Conv2d(128, 128, kernel_size=self.kernel, padding=self.padding, stride=self.stride)
        self._l4_norm = nn.BatchNorm2d(128)
        self._l4_act = nn.ReLU()

        self.l5 = self.conv2d_transpose_replacement(128, 128)
        self._l5_norm = nn.BatchNorm2d(128)
        self._l5_act = nn.ReLU()
        self.l6 = self.conv2d_transpose_replacement(128, 64)
        self._l6_norm = nn.BatchNorm2d(64)
        self._l6_act = nn.ReLU()
        self.l7 = self.conv2d_transpose_replacement(64, 32)
        self._l7_norm = nn.BatchNorm2d(32)
        self._l7_act = nn.ReLU()
        self.l8 = self.conv2d_transpose_replacement(32, self.in_channels)
        self._l8_act = nn.Tanh()

    def forward(self, x):
        assert x.shape[1:] == (self.in_channels, 128, 128)

        x_1 = self._l1_act(self._l1_norm(self.l1(x)))
        x_2 = self._l2_act(self._l2_norm(self.l2(x_1)))
        x_3 = self._l3_act(self._l3_norm(self.l3(x_2)))
        x_4 = self._l4_act(self._l4_norm(self.l4(x_3)))
        x_5 = self._l5_act(self._l5_norm(self.l5(x_4)))
        x_6 = self._l6_act(self._l6_norm(self.l6(x_5)))
        x_7 = self._l7_act(self._l7_norm(self.l7(x_6)))
        x_8 = self._l8_act(self.l8(x_7))

        assert x_8.shape[1:] == (self.in_channels, 128, 128)
        return x_8

    def conv2d_transpose_replacement(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Upsample(scale_factor=self.scale_factor, mode=self.upsample_mode),
            nn.ReflectionPad2d(self.padding_reflection_pad),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=self.kernel_conv2d_replacement,
                stride=self.stride_conv2d_replacement,
                padding=self.padding_conv2d_replacement
            ),
        )


class AEBigNet(nn.Module):

    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.kernel = 3
        self.padding = 1
        self.stride = 2
        self.output_padding = 1

        self.l1 = nn.Conv2d(self.in_channels, 32, kernel_size=self.kernel, padding=self.padding, stride=self.stride)
        self._l1_norm = nn.BatchNorm2d(32)
        self._l1_act = nn.ReLU()
        self.l2 = nn.Conv2d(32, 64, kernel_size=self.kernel, padding=self.padding, stride=self.stride)
        self._l2_norm = nn.BatchNorm2d(64)
        self._l2_act = nn.ReLU()
        self.l3 = nn.Conv2d(64, 128, kernel_size=self.kernel, padding=self.padding, stride=self.stride)
        self._l3_norm = nn.BatchNorm2d(128)
        self._l3_act = nn.ReLU()

        self.l5 = nn.ConvTranspose2d(128, 128, kernel_size=self.kernel, padding=self.padding, stride=self.stride, output_padding=self.output_padding)
        self._l5_norm = nn.BatchNorm2d(128)
        self._l5_act = nn.ReLU()
        self.l6 = nn.ConvTranspose2d(128, 64, kernel_size=self.kernel, padding=self.padding, stride=self.stride, output_padding=self.output_padding)
        self._l6_norm = nn.BatchNorm2d(64)
        self._l6_act = nn.ReLU()
        self.l7 = nn.ConvTranspose2d(64, self.in_channels, kernel_size=self.kernel, padding=self.padding, stride=self.stride, output_padding=self.output_padding)
        self._l7_act = nn.Tanh()

    def forward(self, x):
        assert x.shape[1:] == (self.in_channels, 128, 128)

        x_1 = self._l1_act(self._l1_norm(self.l1(x)))
        x_2 = self._l2_act(self._l2_norm(self.l2(x_1)))
        x_3 = self._l3_act(self._l3_norm(self.l3(x_2)))
        x_5 = self._l5_act(self._l5_norm(self.l5(x_3)))
        x_6 = self._l6_act(self._l6_norm(self.l6(x_5)))
        x_7 = self._l7_act(self.l7(x_6))

        assert x_7.shape[1:] == (self.in_channels, 128, 128)
        return x_7


class AEBiggerNet(nn.Module):

    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.kernel = 3
        self.padding = 1
        self.stride = 2
        self.output_padding = 1

        self.l1 = nn.Conv2d(self.in_channels, 64, kernel_size=self.kernel, padding=self.padding, stride=self.stride)
        self._l1_norm = nn.BatchNorm2d(64)
        self._l1_act = nn.ReLU()
        self.l2 = nn.Conv2d(64, 256, kernel_size=self.kernel, padding=self.padding, stride=self.stride)
        self._l2_norm = nn.BatchNorm2d(256)
        self._l2_act = nn.ReLU()
        self.l3 = nn.Conv2d(256, 512, kernel_size=self.kernel, padding=self.padding, stride=self.stride)
        self._l3_norm = nn.BatchNorm2d(512)
        self._l3_act = nn.ReLU()

        self.l5 = nn.ConvTranspose2d(512, 256, kernel_size=self.kernel, padding=self.padding, stride=self.stride, output_padding=self.output_padding)
        self._l5_norm = nn.BatchNorm2d(256)
        self._l5_act = nn.ReLU()
        self.l6 = nn.ConvTranspose2d(256, 64, kernel_size=self.kernel, padding=self.padding, stride=self.stride, output_padding=self.output_padding)
        self._l6_norm = nn.BatchNorm2d(64)
        self._l6_act = nn.ReLU()
        self.l7 = nn.ConvTranspose2d(64, self.in_channels, kernel_size=self.kernel, padding=self.padding, stride=self.stride, output_padding=self.output_padding)
        self._l7_act = nn.Tanh()

    def forward(self, x):
        assert x.shape[1:] == (self.in_channels, 128, 128)

        x_1 = self._l1_act(self._l1_norm(self.l1(x)))
        x_2 = self._l2_act(self._l2_norm(self.l2(x_1)))
        x_3 = self._l3_act(self._l3_norm(self.l3(x_2)))
        x_5 = self._l5_act(self._l5_norm(self.l5(x_3)))
        x_6 = self._l6_act(self._l6_norm(self.l6(x_5)))
        x_7 = self._l7_act(self.l7(x_6))

        assert x_7.shape[1:] == (self.in_channels, 128, 128)
        return x_7


class AEBiggerNoTransposeNet(nn.Module):

    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.kernel = 3
        self.padding = 1
        self.stride = 2
        self.output_padding = 1

        self.scale_factor = 2
        self.upsample_mode = 'nearest'
        self.padding_reflection_pad = 1
        self.padding_conv2d_replacement = 0
        self.kernel_conv2d_replacement = 3
        self.stride_conv2d_replacement = 1

        self.l1 = nn.Conv2d(self.in_channels, 64, kernel_size=self.kernel, padding=self.padding, stride=self.stride)
        self._l1_norm = nn.BatchNorm2d(64)
        self._l1_act = nn.ReLU()
        self.l2 = nn.Conv2d(64, 256, kernel_size=self.kernel, padding=self.padding, stride=self.stride)
        self._l2_norm = nn.BatchNorm2d(256)
        self._l2_act = nn.ReLU()
        self.l3 = nn.Conv2d(256, 512, kernel_size=self.kernel, padding=self.padding, stride=self.stride)
        self._l3_norm = nn.BatchNorm2d(512)
        self._l3_act = nn.ReLU()

        self.l5 = self.conv2d_transpose_replacement(512, 256)
        self._l5_norm = nn.BatchNorm2d(256)
        self._l5_act = nn.ReLU()
        self.l6 = self.conv2d_transpose_replacement(256, 64)
        self._l6_norm = nn.BatchNorm2d(64)
        self._l6_act = nn.ReLU()
        self.l7 = self.conv2d_transpose_replacement(64, self.in_channels)
        self._l7_act = nn.Tanh()

    def forward(self, x):
        assert x.shape[1:] == (self.in_channels, 128, 128)

        x_1 = self._l1_act(self._l1_norm(self.l1(x)))
        x_2 = self._l2_act(self._l2_norm(self.l2(x_1)))
        x_3 = self._l3_act(self._l3_norm(self.l3(x_2)))
        x_5 = self._l5_act(self._l5_norm(self.l5(x_3)))
        x_6 = self._l6_act(self._l6_norm(self.l6(x_5)))
        x_7 = self._l7_act(self.l7(x_6))

        assert x_7.shape[1:] == (self.in_channels, 128, 128)
        return x_7

    def conv2d_transpose_replacement(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Upsample(scale_factor=self.scale_factor, mode=self.upsample_mode),
            nn.ReflectionPad2d(self.padding_reflection_pad),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=self.kernel_conv2d_replacement,
                stride=self.stride_conv2d_replacement,
                padding=self.padding_conv2d_replacement
            ),
        )


class AESmallNoTransposeNet(nn.Module):

    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.kernel = 3
        self.padding = 1
        self.stride = 2
        self.output_padding = 1

        self.scale_factor = 2
        self.upsample_mode = 'nearest'
        self.padding_reflection_pad = 1
        self.padding_conv2d_replacement = 0
        self.kernel_conv2d_replacement = 3
        self.stride_conv2d_replacement = 1

        self.l1 = nn.Conv2d(self.in_channels, 128, kernel_size=self.kernel, padding=self.padding, stride=self.stride)
        self._l1_norm = nn.BatchNorm2d(128)
        self._l1_act = nn.ReLU()
        self.l2 = nn.Conv2d(128, 512, kernel_size=self.kernel, padding=self.padding, stride=self.stride)
        self._l2_norm = nn.BatchNorm2d(512)
        self._l2_act = nn.ReLU()

        self.l6 = self.conv2d_transpose_replacement(512, 128)
        self._l6_norm = nn.BatchNorm2d(128)
        self._l6_act = nn.ReLU()
        self.l7 = self.conv2d_transpose_replacement(128, self.in_channels)
        self._l7_act = nn.Tanh()

    def forward(self, x):
        assert x.shape[1:] == (self.in_channels, 128, 128)

        x_1 = self._l1_act(self._l1_norm(self.l1(x)))
        x_2 = self._l2_act(self._l2_norm(self.l2(x_1)))
        x_6 = self._l6_act(self._l6_norm(self.l6(x_2)))
        x_7 = self._l7_act(self.l7(x_6))

        assert x_7.shape[1:] == (self.in_channels, 128, 128)
        return x_7

    def conv2d_transpose_replacement(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Upsample(scale_factor=self.scale_factor, mode=self.upsample_mode),
            nn.ReflectionPad2d(self.padding_reflection_pad),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=self.kernel_conv2d_replacement,
                stride=self.stride_conv2d_replacement,
                padding=self.padding_conv2d_replacement
            ),
        )


class AEDhurandhar(nn.Module):

    def __init__(self, in_channels=1):
        super().__init__()
        self.in_channels = in_channels

        self.m = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        self.up = nn.Upsample(scale_factor=2)

        self.n = nn.Sequential(

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        assert x.shape[1:] == (self.in_channels, 128, 128)

        x = self.m(x)
        x = self.up(x)
        x = self.n(x)

        assert x.shape[1:] == (self.in_channels, 128, 128)
        return x