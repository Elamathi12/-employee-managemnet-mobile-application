import torch
import torch.nn as nn
import numpy as np


class Generator(nn.Module):
    def __init__(self, img_shape, latent_dim):
        super(Generator, self).__init__()
        
        self.fc1 = nn.Linear(latent_dim, 128)
        self.bn1 = nn.BatchNorm1d(128, 0.8)
        self.fc2 = nn.Linear(128, 256)
        self.bn2 = nn.BatchNorm1d(256, 0.8)
        self.fc3 = nn.Linear(256, 512)
        self.bn3 = nn.BatchNorm1d(512, 0.8)
        self.fc4 = nn.Linear(512, 1024)
        self.bn4 = nn.BatchNorm1d(1024, 0.8)
        self.fc5 = nn.Linear(1024, int(np.prod(img_shape)))
        self.tanh = nn.Tanh()
        self.relu = nn.LeakyReLU(0.2,inplace=True)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.fc1.weight.data, 0.0, 0.02)
        nn.init.normal_(self.fc2.weight.data, 0.0, 0.02)
        nn.init.normal_(self.fc3.weight.data, 0.0, 0.02)
        nn.init.normal_(self.fc4.weight.data, 0.0, 0.02)
        nn.init.normal_(self.fc5.weight.data, 0.0, 0.02)

    def to_img(x):
        x = x.clamp(0, 1)
        return x

    def forward(self,img_shape, z):
        x = self.relu(self.bn1(self.fc1(z)))
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.relu(self.bn3(self.fc3(x)))
        x = self.relu(self.bn4(self.fc4(x)))
        x = self.tanh(self.fc5(x))
        img = x.view(x.size(0), *img_shape)
        return img


