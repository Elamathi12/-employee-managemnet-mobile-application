import torch
import torch.nn as nn
from trial import *
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.init_size = 28 // 4
        self.hidden_channel = 128

        self.l1 = nn.Linear(64, 128)
        self.leaky_relu1 = nn.LeakyReLU(0.2, inplace=True)

        self.l2 = nn.Linear(128, 256)
        self.leaky_relu2 = nn.LeakyReLU(0.2, inplace=True)

        self.l3 = nn.Linear(256, self.hidden_channel * self.init_size ** 2)
        self.leaky_relu3 = nn.LeakyReLU(0.2, inplace=True)

        self.bn1 = nn.BatchNorm2d(self.hidden_channel)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv1 = nn.Conv2d(self.hidden_channel, self.hidden_channel, 3, stride=1, padding=1, padding_mode='reflect')
        self.bn2 = nn.BatchNorm2d(self.hidden_channel, 0.8)
        self.leaky_relu4 = nn.LeakyReLU(0.2, inplace=True)

        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv2 = nn.Conv2d(self.hidden_channel, self.hidden_channel//2, 3, stride=1, padding=1, padding_mode='reflect')
        self.bn3 = nn.BatchNorm2d(self.hidden_channel//2, 0.8)
        self.leaky_relu5 = nn.LeakyReLU(0.2, inplace=True)

        self.conv3 = nn.Conv2d(self.hidden_channel//2, self.hidden_channel//2, 3, stride=1, padding=1, padding_mode='reflect')
        self.bn4 = nn.BatchNorm2d(self.hidden_channel//2, 0.8)
        self.leaky_relu6 = nn.LeakyReLU(0.2, inplace=True)

        self.conv4 = nn.Conv2d(self.hidden_channel//2, 1, 3, stride=1, padding=1, padding_mode='reflect')

    def forward(self, z):
        out = self.l1(z)
        out = self.leaky_relu1(out)

        out = self.l2(out)
        out = self.leaky_relu2(out)

        out = self.l3(out)
        out = self.leaky_relu3(out)

        out = out.view(out.shape[0], self.hidden_channel, self.init_size, self.init_size)

        out = self.bn1(out)
        out = self.upsample1(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.leaky_relu4(out)

        out = self.upsample2(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.leaky_relu5(out)

        out = self.conv3(out)
        out = self.bn4(out)
        out = self.leaky_relu6(out)

        img = self.conv4(out)

        return img
