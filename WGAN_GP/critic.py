import torch
import torch.nn as nn
import torch.nn.functional as F


class Critic(nn.Module):
    def __init__(self, input_shape):
        super(Critic, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 64, kernel_size=4, stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.LeakyReLU(0.2,inplace=True)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.LeakyReLU(0.2,inplace=True)
        self.conv3 = nn.Conv2d(128, 1, kernel_size=4, stride=2)

    def forward(self, input_shape):
        print(input_shape.shape,"shape of input_shape")
        x = self.relu1(self.bn1(self.conv1(input_shape)))
        print(x.shape, "1")
        x = self.relu2(self.bn2(self.conv2(x)))
        print(x.shape, "2")
        x = self.conv3(x)
        print(x.shape, "3")
        x = x.view(len(x), -1)
        print(x.shape, "4")

        return x 