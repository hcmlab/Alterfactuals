import torch.nn as nn


class G(nn.Module):

    def __init__(self, in_channels=1, out_channels=1, input_size=128,):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_size = input_size

        self.kernel = 3
        self.stride = 2

        self.dropout = 0.5

        self.l1, self.l1_act = self.encoder_block(1, 64)
        self.l2, self.l2_act = self.encoder_block(64, 256)
        self.l3, self.l3_act = self.encoder_block(256, 512)
        self.l4, self.l4_act = self.encoder_block(512, 512)

        self.l5, self.l5_act = self.decoder_block(512, 512, dropout=True)
        self.l6, self.l6_act = self.decoder_block(512, 256, dropout=True)
        self.l7, self.l7_act = self.decoder_block(256, 64, dropout=True)

        self.l8 = nn.Sequential(
            nn.ConvTranspose2d(64, 1, kernel_size=self.kernel, stride=self.stride, output_padding=1),
            nn.Tanh()
        )

    def encoder_block(self, in_channels, out_channels, batch_norm=True):
        if batch_norm:
            model = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=self.kernel, stride=self.stride),
                nn.BatchNorm2d(out_channels)
            )
        else:
            model = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=self.kernel, stride=self.stride)
            )

        return model, nn.LeakyReLU()

    def decoder_block(self, in_channels, out_channels, dropout=False):
        main_layer = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=self.kernel, stride=self.stride)

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

        return model, nn.LeakyReLU()

    def forward(self, x, labels):
        self.assert_size(x, self.in_channels, self.input_size)

        l1_raw = self.l1(x)
        l1_act = self.l1_act(l1_raw)

        l2_raw = self.l2(l1_act)
        l2_act = self.l2_act(l2_raw)

        l3_raw = self.l3(l2_act)
        l3_act = self.l3_act(l3_raw)

        l4_raw = self.l4(l3_act)
        l4_act = self.l4_act(l4_raw)

        l5_raw = self.l5(l4_act)
        l5_act = self.l5_act(l5_raw)

        l6_raw = self.l6(l5_act)
        l6_act = self.l6_act(l6_raw)

        l7_raw = self.l7(l6_act)
        l7_act = self.l7_act(l7_raw)

        output = self.l8(l7_act)

        self.assert_size(output, self.out_channels, self.input_size)
        return output

    def assert_size(self, tensor, channel, size):
        assert tensor.shape[1:] == (channel, size, size)