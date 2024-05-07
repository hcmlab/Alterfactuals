import torch
from torch import nn

# paper: Isola


class GPix2Pix(nn.Module):

    def __init__(self, in_channels=1, out_channels=1, input_size=256, use_conv2dtranspose=True, upsample_mode='nearest'):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_size = input_size

        self.use_conv2dtranspose = use_conv2dtranspose

        self.kernel = 4
        self.stride = 2
        self.padding_conv2d = 1

        self.scale_factor = 2
        self.upsample_mode = upsample_mode
        self.padding_reflection_pad = 1
        self.padding_conv2d_replacement = 0
        self.kernel_conv2d_replacement = 3
        self.stride_conv2d_replacement = 1

        self.dropout = 0.5

        self.leaky_slope = 0.02

        self.e1, self.e1_act = self.encoder_block(self.in_channels, 64, batch_norm=False)
        self.e2, self.e2_act = self.encoder_block(64, 128)
        self.e3, self.e3_act = self.encoder_block(128, 256)
        self.e4, self.e4_act = self.encoder_block(256, 512)
        self.e5, self.e5_act = self.encoder_block(512, 512)
        self.e6, self.e6_act = self.encoder_block(512, 512)
        self.e7, self.e7_act = self.encoder_block(512, 512)

        self.middle = nn.Conv2d(512, 512, kernel_size=self.kernel, stride=self.stride, padding=self.padding_conv2d)
        self.middle_act = nn.ReLU()

        self.d1, self.d1_act = self.decoder_block(512, 512, dropout=True)
        self.d2, self.d2_act = self.decoder_block(1024, 512, dropout=True)
        self.d3, self.d3_act = self.decoder_block(1024, 512, dropout=True)
        self.d4, self.d4_act = self.decoder_block(1024, 512)
        self.d5, self.d5_act = self.decoder_block(1024, 256)
        self.d6, self.d6_act = self.decoder_block(512, 128)
        self.d7, self.d7_act = self.decoder_block(256, 64)

        self.output = nn.Sequential(
            nn.ConvTranspose2d(128, 1, kernel_size=self.kernel, stride=self.stride, padding=self.padding_conv2d),
            nn.Tanh()
        )

    def encoder_block(self, in_channels, out_channels, batch_norm=True):
        if batch_norm:
            model = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=self.kernel, stride=self.stride, padding=self.padding_conv2d),
                nn.BatchNorm2d(out_channels)
            )
        else:
            model = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=self.kernel, stride=self.stride, padding=self.padding_conv2d)
            )

        return model, nn.LeakyReLU(negative_slope=self.leaky_slope)

    def decoder_block(self, in_channels, out_channels, dropout=False):
        if self.use_conv2dtranspose:
            main_layer = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=self.kernel, stride=self.stride, padding=self.padding_conv2d)
        else:
            main_layer = self.conv2d_transpose_replacement(in_channels, out_channels)

        if dropout:
            model = nn.Sequential(
                main_layer,
                nn.BatchNorm2d(out_channels),
                nn.Dropout(p=self.dropout)
            )
        else:
            model = nn.Sequential(
                main_layer,
                nn.BatchNorm2d(out_channels),
            )

        return model, nn.ReLU()

    # https: // distill.pub / 2016 / deconv - checkerboard /
    # https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/issues/190
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

    def forward(self, x):
        self.assert_size(x, self.in_channels, self.input_size)

        e1_raw = self.e1(x)
        e1_out = self.e1_act(e1_raw)

        e2_raw = self.e2(e1_out)
        e2_out = self.e2_act(e2_raw)

        e3_raw = self.e3(e2_out)
        e3_out = self.e3_act(e3_raw)

        e4_raw = self.e4(e3_out)
        e4_out = self.e4_act(e4_raw)

        e5_raw = self.e5(e4_out)
        e5_out = self.e5_act(e5_raw)

        e6_raw = self.e6(e5_out)
        e6_out = self.e6_act(e6_raw)

        e7_raw = self.e7(e6_out)
        e7_out = self.e7_act(e7_raw)

        middle_raw = self.middle(e7_out)
        middle_out = self.middle_act(middle_raw)
        self.assert_size(middle_out, 512, 1)

        d1_raw = self.d1(middle_out)
        d1_concat = torch.cat([d1_raw, e7_raw], 1)
        d1_out = self.d1_act(d1_concat)

        d2_raw = self.d2(d1_out)
        d2_concat = torch.cat([d2_raw, e6_raw], 1)
        d2_out = self.d2_act(d2_concat)

        d3_raw = self.d3(d2_out)
        d3_concat = torch.cat([d3_raw, e5_raw], 1)
        d3_out = self.d3_act(d3_concat)

        d4_raw = self.d4(d3_out)
        d4_concat = torch.cat([d4_raw, e4_raw], 1)
        d4_out = self.d4_act(d4_concat)

        d5_raw = self.d5(d4_out)
        d5_concat = torch.cat([d5_raw, e3_raw], 1)
        d5_out = self.d5_act(d5_concat)

        d6_raw = self.d6(d5_out)
        d6_concat = torch.cat([d6_raw, e2_raw], 1)
        d6_out = self.d6_act(d6_concat)

        d7_raw = self.d7(d6_out)
        d7_concat = torch.cat([d7_raw, e1_raw], 1)
        d7_out = self.d7_act(d7_concat)

        output_out = self.output(d7_out)
        self.assert_size(output_out, self.out_channels, self.input_size)

        return output_out

    def assert_size(self, tensor, channel, size):
        assert tensor.shape[1:] == (channel, size, size)
