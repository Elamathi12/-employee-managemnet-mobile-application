import torch
import torch.nn as nn
import torch.nn.functional as F


class Critic(nn.Module):
    def __init__(self, im_chan=1, hidden_dim=64):
        super(Critic, self).__init__()
        self.conv1 = nn.Conv2d(im_chan, hidden_dim, kernel_size=4, stride=2)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.relu1 = nn.LeakyReLU(0.2,inplace=True)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(hidden_dim * 2)
        self.relu2 = nn.LeakyReLU(0.2,inplace=True)
        self.conv3 = nn.Conv2d(hidden_dim * 2, 1, kernel_size=4, stride=2)

    def forward(self, image):
    
        x = self.relu1(self.bn1(self.conv1(image)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.conv3(x)
        return x.view(len(x), -1)  # Flatten the output