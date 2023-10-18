import torch
import torch.nn as nn
from trial import * 
from Utils import *



class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        
        self.hidden_channel = 16
        
        self.conv1 = nn.Conv2d(1, self.hidden_channel, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        self.leaky_relu1 = nn.LeakyReLU(0.2, inplace=True)
        self.avgpool1 = nn.AvgPool2d(2, stride=2)
        self.dropout1 = nn.Dropout2d(0.2)

        self.conv2 = nn.Conv2d(self.hidden_channel, self.hidden_channel*2, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        self.leaky_relu2 = nn.LeakyReLU(0.2, inplace=True)
        self.avgpool2 = nn.AvgPool2d(2, stride=2)
        self.dropout2 = nn.Dropout2d(0.2)

        self.conv3 = nn.Conv2d(self.hidden_channel*2, self.hidden_channel*4, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        self.leaky_relu3 = nn.LeakyReLU(0.2, inplace=True)
        self.avgpool3 = nn.AvgPool2d(2, stride=2)
        self.dropout3 = nn.Dropout2d(0.2)

        self.conv4 = nn.Conv2d(self.hidden_channel*4, self.hidden_channel*8, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        self.leaky_relu4 = nn.LeakyReLU(0.2, inplace=True)
        self.avgpool4 = nn.AvgPool2d(2, stride=2)
        self.dropout4 = nn.Dropout2d(0.2)

        # The height and width of downsampled image
        ds_size = img_size // 2 ** 4
        self.minibatch_discrimination = MinibatchDiscrimination(self.hidden_channel*8 * ds_size ** 2, 32, 32)
        self.fc = nn.Linear(32 + self.hidden_channel*8 * ds_size ** 2, 1)

    def forward(self, img):
        out = self.conv1(img)
        out = self.leaky_relu1(out)
        out = self.avgpool1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.leaky_relu2(out)
        out = self.avgpool2(out)
        out = self.dropout2(out)

        out = self.conv3(out)
        out = self.leaky_relu3(out)
        out = self.avgpool3(out)
        out = self.dropout3(out)

        out = self.conv4(out)
        out = self.leaky_relu4(out)
        out = self.avgpool4(out)
        out = self.dropout4(out)

        out = out.view(out.shape[0], -1)
        out = self.minibatch_discrimination(out)
        validity = self.fc(out)

        return validity
