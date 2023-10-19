import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.l1 = nn.Linear(64, 128)
        self.leaky_relu1 = nn.LeakyReLU(0.2, inplace=True)

        self.l2 = nn.Linear(128, 256)
        self.leaky_relu2 = nn.LeakyReLU(0.2, inplace=True)

        self.l3 = nn.Linear(256, 6272)
        self.leaky_relu3 = nn.LeakyReLU(0.2, inplace=True)

        self.bn1 = nn.BatchNorm2d(128)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv1 = nn.Conv2d(128, 128, 3, stride=1, padding=1, padding_mode='reflect')
        self.bn2 = nn.BatchNorm2d(128, 0.8)
        self.leaky_relu4 = nn.LeakyReLU(0.2, inplace=True)

        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv2 = nn.Conv2d(128, 64, 3, stride=1, padding=1, padding_mode='reflect')
        self.bn3 = nn.BatchNorm2d(64, 0.8)
        self.leaky_relu5 = nn.LeakyReLU(0.2, inplace=True)

        self.conv3 = nn.Conv2d(64, 64, 3, stride=1, padding=1, padding_mode='reflect')
        self.bn4 = nn.BatchNorm2d(64, 0.8)
        self.leaky_relu6 = nn.LeakyReLU(0.2, inplace=True)

        self.conv4 = nn.Conv2d(64, 1, 3, stride=1, padding=1, padding_mode='reflect')

    def forward(self, z):
        x = self.leaky_relu1(self.l1(z))
        x = self.leaky_relu2(self.l2(x))
        x = self.leaky_relu3(self.l3(x))
        x = x.view(x.shape[0], 128, 7,7)
        x = self.bn1(x)
        x = self.upsample1(x)
        x = self.leaky_relu4(self.bn2(self.conv1(x)))
        x = self.upsample2(x)
        x = self.leaky_relu5(self.bn3(self.conv2(x)))
        x = self.leaky_relu6(self.bn4(self.conv3(x)))
        img = self.conv4(x)
    
        return img
