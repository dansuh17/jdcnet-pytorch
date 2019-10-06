"""
Implementation of model from:

Kum et al. - "Joint Detection and Classification of Singing Voice Melody Using
Convolutional Recurrent Neural Networks" (2019)

Link: https://www.semanticscholar.org/paper/Joint-Detection-and-Classification-of-Singing-Voice-Kum-Nam/60a2ad4c7db43bace75805054603747fcd062c0d
"""
import torch
from torch import nn


class JDCNet(nn.Module):
    """
    Joint Detection and Classification Network model for singing voice melody.
    """
    def __init__(self, num_class=722, seq_len=31, leaky_relu_slope=0.01):
        super().__init__()
        self.seq_len = seq_len  # 31
        self.num_class = num_class

        # input = (b, 1, 31, 513), b = batch size
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1, bias=False),  # out: (b, 64, 31, 513)
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(leaky_relu_slope, inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),  # (b, 64, 31, 513)

            # res blocks
            ResBlock(in_channels=64, out_channels=128),  # (b, 128, 31, 128)
            ResBlock(in_channels=128, out_channels=192),  # (b, 192, 31, 32)
            ResBlock(in_channels=192, out_channels=256),  # (b, 256, 31, 8)

            # pool block
            nn.BatchNorm2d(num_features=256),
            nn.LeakyReLU(leaky_relu_slope, inplace=True),
            nn.MaxPool2d(kernel_size=(1, 4)),  # (b, 256, 31, 2)
            nn.Dropout(p=0.5),
        )

        # input: (31, b, 512) - resized from (b, 256, 31, 2)
        self.lstm = nn.LSTM(input_size=512, hidden_size=256, dropout=0.3, bidirectional=True)  # (b, 31, 512)

        # input: (b * 31, 512)
        self.classifier = nn.Linear(in_features=512, out_features=self.num_class)  # (b * 31, num_class)

    def forward(self, x):
        x = self.net(x)

        x = x.view((-1, 31, 512))  # (b, 31, 512)
        x.permute(1, 0, 2)  # (31, b, 512) - batch and sequence length swapped
        x, _ = self.lstm(x)  # ignore the hidden states

        x = x.view((-1, 512))  # (b * 31, 512)
        x = self.classifier(x)
        class_logit = x.view((-1, 31, self.num_class))

        detect_v = None  # TODO:
        return class_logit, detect_v


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, leaky_relu_slope=0.01):
        super().__init__()
        self.downsample = in_channels != out_channels

        # BN / LReLU / MaxPool layer before the conv layer - see Figure 1b in the paper
        self.pre_conv = nn.Sequential(
            nn.BatchNorm2d(num_features=in_channels),
            nn.LeakyReLU(leaky_relu_slope, inplace=True),
            nn.MaxPool2d(kernel_size=(1, 4)),  # apply downsampling on the y axis only
        )

        # conv layers
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(leaky_relu_slope, inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
        )

        # 1 x 1 convolution layer to match the feature dimensions
        self.conv1by1 = None
        if self.downsample:
            self.conv1by1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)

    def forward(self, x):
        x = self.pre_conv(x)
        if self.downsample:
            x = self.conv(x) + self.conv1by1(x)
        else:
            x = self.conv(x) + x
        return x


if __name__ == '__main__':
    dummy = torch.randn((10, 1, 31, 513))  # dummy random input
    jdc = JDCNet()
    clss, detect = jdc(dummy)
    print(clss.size())
