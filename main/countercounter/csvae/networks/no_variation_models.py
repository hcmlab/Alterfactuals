import torch
from torch import nn


class EncoderXToZNoVar(nn.Module):

    def __init__(self, input_size, z_size):
        super().__init__()

        self.input_size = input_size
        self.z_size = z_size

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(16 * 16 * 64, 2048),
            nn.BatchNorm1d(2048),
            nn.LeakyReLU(),
            nn.Linear(2048, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(128, self.z_size),
            nn.BatchNorm1d(self.z_size),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        encoding = self.encoder(x)

        return encoding


class EncoderXToZNoVarConv(nn.Module):

    def __init__(self, input_size, z_size=None):
        super().__init__()

        self.input_size = input_size

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
        )  # size: n, 64, 16, 16

    def forward(self, x):
        encoding = self.encoder(x)

        return encoding


class EncoderYToWNoVar(nn.Module):

    def __init__(self, label_size, w_size):
        super().__init__()
        self.label_size = label_size
        self.w_size = w_size

        self.encoder = nn.Sequential(
            nn.Linear(self.label_size, self.w_size),
            nn.LeakyReLU(),
            nn.Linear(self.w_size, self.w_size),
            nn.LeakyReLU(),
        )

    def forward(self, y):
        encoding = self.encoder(y.view(-1, 1).float())
        return encoding


class EncoderYToWNoVarConv(nn.Module):

    def __init__(self, label_size, w_size):
        super().__init__()
        self.label_size = label_size
        self.w_size = w_size

        self.pre_encoder = nn.Sequential(
            nn.Linear(self.label_size, 256),
            nn.LeakyReLU(),
        )
        self.encoder = nn.Sequential(
            nn.ConvTranspose2d(64, 256, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=2, output_padding=1),
            nn.LeakyReLU(),
        )

    def forward(self, y):
        pre_encoding = self.pre_encoder(y.view(-1, 1).float())
        pre_encoding = pre_encoding.view(-1, 64, 2, 2)
        encoding = self.encoder(pre_encoding)
        return encoding

class EncoderXYToWNoVar(nn.Module):

    def __init__(self, input_size, label_size, w_size):
        super().__init__()

        self.input_size = input_size
        self.label_size = label_size
        self.w_size = w_size

        self.emb_size = 8
        self.emb_layer = nn.Embedding(2, self.emb_size * self.emb_size)
        self.upsample_layer = nn.Upsample(size=(self.input_size, self.input_size))

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(16 * 16 * 64, 2048),
            nn.BatchNorm1d(2048),
            nn.LeakyReLU(),
            nn.Linear(2048, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(128, self.w_size),
            nn.BatchNorm1d(self.w_size),
            nn.LeakyReLU(),
        )

    def forward(self, x, y):
        labels_embedded = self.emb_layer(y)
        labels_embedded = labels_embedded.reshape(labels_embedded.size(0), 1, self.emb_size, self.emb_size)
        labels_upsampled = self.upsample_layer(labels_embedded)

        x = torch.cat([x, labels_upsampled], dim=1)

        encoding = self.encoder(x)

        return encoding


class EncoderXYToWLessClassNoVar(nn.Module):

    def __init__(self, input_size, label_size, w_size):
        super().__init__()

        self.input_size = input_size
        self.label_size = label_size
        self.w_size = w_size

        self.emb_size = 8
        self.emb_layer = nn.Embedding(2, self.emb_size * self.emb_size)
        self.upsample_layer = nn.Upsample(size=(16, 16))

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
        )
        self.enc_with_class = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 16 * 65, 2048),
            nn.BatchNorm1d(2048),
            nn.LeakyReLU(),
            nn.Linear(2048, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(128, self.w_size),
            nn.BatchNorm1d(self.w_size),
            nn.LeakyReLU(),
        )


    def forward(self, x, y):
        labels_embedded = self.emb_layer(y)
        labels_embedded = labels_embedded.reshape(labels_embedded.size(0), 1, self.emb_size, self.emb_size)
        labels_upsampled = self.upsample_layer(labels_embedded)

        encoding = self.encoder(x)

        xy = torch.cat([encoding, labels_upsampled], dim=1)

        encoding_with_class = self.enc_with_class(xy)

        return encoding_with_class


class EncoderXYToWEvenLessClassNoVar(nn.Module):

    def __init__(self, input_size, label_size, w_size):
        super().__init__()

        self.input_size = input_size
        self.label_size = label_size
        self.w_size = w_size

        self.emb_size = 8
        self.emb_layer = nn.Embedding(2, self.emb_size * self.emb_size)
        self.upsample_layer = nn.Upsample(size=(16, 16))

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(16 * 16 * 64, 2048),
            nn.BatchNorm1d(2048),
            nn.LeakyReLU(),
            nn.Linear(2048, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
        )
        self.enc_with_class = nn.Sequential(
            nn.Linear(65, self.w_size),
            nn.BatchNorm1d(self.w_size),
            nn.LeakyReLU(),
        )


    def forward(self, x, y):
        encoding = self.encoder(x)

        xy = torch.cat([encoding, y.view(-1, 1)], dim=1)

        encoding_with_class = self.enc_with_class(xy)

        return encoding_with_class


class EncoderXYToWNoVarConv(nn.Module):

    def __init__(self, input_size, label_size, w_size=None):
        super().__init__()

        self.input_size = input_size
        self.label_size = label_size

        self.emb_size = 8
        self.emb_layer = nn.Embedding(2, self.emb_size * self.emb_size)
        self.upsample_layer = nn.Upsample(size=(self.input_size, self.input_size))

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
        ) # size: n, 64, 16, 16

    def forward(self, x, y):
        labels_embedded = self.emb_layer(y)
        labels_embedded = labels_embedded.reshape(labels_embedded.size(0), 1, self.emb_size, self.emb_size)
        labels_upsampled = self.upsample_layer(labels_embedded)

        x = torch.cat([x, labels_upsampled], dim=1)

        encoding = self.encoder(x)

        return encoding


class EncoderXYToWNoClassNoVar(nn.Module):

    def __init__(self, input_size, label_size, w_size):
        super().__init__()

        self.input_size = input_size
        self.label_size = label_size
        self.w_size = w_size

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(16 * 16 * 64, 2048),
            nn.BatchNorm1d(2048),
            nn.LeakyReLU(),
            nn.Linear(2048, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
        )
        self.enc_with_class = nn.Sequential(
            nn.Linear(64, self.w_size),
            nn.BatchNorm1d(self.w_size),
            nn.LeakyReLU(),
        )

    def forward(self, x, y):
        encoding = self.encoder(x)

        encoding_with_class = self.enc_with_class(encoding)

        return encoding_with_class


class DecoderZWToXNoVar(nn.Module):

    def __init__(self, z_size, w_size, output_channel):
        super().__init__()

        self.z_size = z_size
        self.w_size = w_size
        self.output_channel = output_channel

        self.pre_decoder = nn.Sequential(
            nn.Linear(self.z_size + self.w_size, self.z_size + self.w_size),
            nn.BatchNorm1d(self.z_size + self.w_size),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.z_size + self.w_size, 512, kernel_size=5, stride=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(256, 256, kernel_size=3, stride=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
        )
        self.output = nn.Sequential(
            nn.ConvTranspose2d(64, self.output_channel, kernel_size=2, stride=2, padding=3),
            nn.Tanh()
        )

    def forward(self, z, w):
        pre_decoding = self.pre_decoder(torch.cat((z, w), dim=1))
        pre_decoding = pre_decoding.view(-1, self.z_size + self.w_size, 1, 1)
        decoding = self.decoder(pre_decoding)

        x_pred = self.output(decoding)
        return x_pred


class DecoderZWToXOtherNoVar(nn.Module):

    def __init__(self, z_size, w_size, output_channel):
        super().__init__()

        self.z_size = z_size
        self.w_size = w_size
        self.output_channel = output_channel

        self.pre_decoder = nn.Sequential(
            nn.Linear(self.z_size + self.w_size, self.z_size + self.w_size),
            nn.BatchNorm1d(self.z_size + self.w_size),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.z_size + self.w_size, 512, kernel_size=5, stride=2, padding=0),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=5, stride=2, padding=0),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(256, 256, kernel_size=5, stride=2, padding=0),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2, padding=0, output_padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
        )
        self.output = nn.Sequential(
            nn.ConvTranspose2d(128, self.output_channel, kernel_size=5, stride=2, padding=0, output_padding=1),
            nn.Tanh()
        )

    def forward(self, z, w):
        pre_decoding = self.pre_decoder(torch.cat((z, w), dim=1))
        pre_decoding = pre_decoding.view(-1, self.z_size + self.w_size, 1, 1)
        decoding = self.decoder(pre_decoding)

        x_pred = self.output(decoding)
        return x_pred


class DecoderZWToXOtherNoVarConv(nn.Module):

    def __init__(self, z_size, w_size, output_channel):
        super().__init__()

        self.z_size = z_size
        self.w_size = w_size
        self.output_channel = output_channel

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64 + 64, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
        )
        self.output = nn.Sequential(
            nn.ConvTranspose2d(128, self.output_channel, kernel_size=3, stride=2, padding=0, output_padding=1),
            nn.Tanh()
        )

    def forward(self, z, w):
        decoding = self.decoder(torch.cat((z, w), dim=1))

        x_pred = self.output(decoding)
        return x_pred


class DecoderZToYNoVar(nn.Module):

    def __init__(self, z_size, label_size):
        super().__init__()

        self.z_size = z_size
        self.label_size = label_size

        self.decoder = nn.Sequential(
            nn.Linear(self.z_size, self.z_size),
            nn.BatchNorm1d(self.z_size),
            nn.LeakyReLU(),
            nn.Linear(self.z_size, self.z_size),
            nn.BatchNorm1d(self.z_size),
            nn.LeakyReLU(),
            nn.Linear(self.z_size, self.label_size),
            nn.BatchNorm1d(self.label_size),
            nn.Softmax(),
        )

    def forward(self, z):
        y_pred = self.decoder(z)
        return y_pred


class DecoderZToYNoVarConv(nn.Module):

    def __init__(self, z_size, label_size):
        super().__init__()

        self.z_size = z_size
        self.label_size = label_size

        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(2 * 2 * 64, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(128, self.label_size),
            nn.BatchNorm1d(self.label_size),
            nn.Softmax(),
        )

    def forward(self, z):
        y_pred = self.decoder(z)
        return y_pred