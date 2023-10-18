import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

# Generator
class Generator(nn.Module):
    def __init__(self, z_dim=10, im_chan=1, hidden_dim=64):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.hidden_dim = 64

        # Generator layers
        self.fc1 = nn.Linear(z_dim, hidden_dim * 4 )
        self.conv1 = nn.ConvTranspose2d(hidden_dim * 4, hidden_dim * 2, kernel_size=4, stride=1)
        self.bn2 = nn.BatchNorm2d(hidden_dim * 2)
        self.conv2 = nn.ConvTranspose2d(hidden_dim * 2, hidden_dim, kernel_size=4, stride=2)
        self.bn3 = nn.BatchNorm2d(hidden_dim)
        self.conv3 = nn.ConvTranspose2d(hidden_dim, im_chan, kernel_size=4, stride=2)

    def forward(self, noise):        
        x = F.relu(self.fc1(noise))        
        x = x.view(len(noise), self.hidden_dim * 4,1,1)  # Reshape
        x = F.relu(self.bn2(self.conv1(x)))
        x = F.relu(self.bn3(self.conv2(x)))
        x = torch.tanh(self.conv3(x))  # Tanh activation for the final layer
        return x