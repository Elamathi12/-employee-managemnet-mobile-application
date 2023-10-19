import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

# Generator
class Generator(nn.Module):
    def __init__(self, input_shape):
        super(Generator, self).__init__()
    
        # Generator layers
        self.fc1 = nn.Linear(input_shape[2], 256 )
        self.conv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv3 = nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2)

    def forward(self, noise):
        print(noise.shape, "shape of noise")        
        x = F.relu(self.fc1(noise)) 
        print(len(noise),"len of noise")       
        x = x.view(len(noise), 256,1,1)  # Reshape
        print(x.shape, "1")
        x = F.relu(self.bn2(self.conv1(x)))
        print(x.shape, "2")
        x = F.relu(self.bn3(self.conv2(x)))
        print(x.shape, "3")
        x = torch.tanh(self.conv3(x))
        print(x.shape, "4")  # Tanh activation for the final layer
        return x
    




    