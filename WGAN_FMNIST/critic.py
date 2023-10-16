import torch
import torch.nn as nn
import numpy as np

class Critic(nn.Module):
    def __init__(self, img_shape):
        super(Critic, self).__init__()

        self.fc1 = nn.Linear(int(np.prod(img_shape)), 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.fc1.weight.data, 0.0, 0.02)
        nn.init.normal_(self.fc2.weight.data, 0.0, 0.02)
        nn.init.normal_(self.fc3.weight.data, 0.0, 0.02)

    def to_img(x):
        x = x.clamp(0, 1)
        return x    
    
    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        x = self.leaky_relu(self.fc1(img_flat))
        x = self.leaky_relu(self.fc2(x)) 
        x = self.fc3(x)
        return x


