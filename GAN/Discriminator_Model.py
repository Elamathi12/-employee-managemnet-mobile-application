import torch 
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self,input_shape): # INPUT : 100 X 1 X 1000
        super(Discriminator, self).__init__()

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ TRANSPOSE CONVOLUTIONAL LAYERS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

        # INPUT : 100 X 1 X 1000
        self.conv1 = nn.Conv1d(in_channels=1 ,out_channels=8,kernel_size=4, stride=2, padding=1)
        self.Instancenorm1 = nn.BatchNorm1d(8)
        # OUTPUT : 100 X 8 X 500
        
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ CONVOLUTIONAL LAYERS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

        # INPUT : 100 X 8 X 500
        self.conv2 = nn.Conv1d(in_channels=8 ,out_channels=16,kernel_size=4, stride=2, padding=1)
        self.Instancenorm2 = nn.BatchNorm1d(16)
        # OUTPUT : 100 X 16 X 250

        # INPUT : 100 X 16 X 250
        self.conv3 = nn.Conv1d(in_channels=16 ,out_channels=1,kernel_size=2, stride=2, padding=1)
        self.Instancenorm3 = nn.BatchNorm1d(1)
        # OUTPUT : 100 X 1 X 126

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ FULLY CONNECTED LAYERS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

        self.fc1 = nn.Linear(in_features=126,out_features=50)
        self.fc2 = nn.Linear(in_features=50,out_features=1)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ACTIVATION LAYER ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

        self.leakyrelu = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()


    def forward(self,x):
        print("~~~~~~~~~Entered forward function in discriminator model~~~~~~~~~~~~") 
        x = self.leakyrelu(self.Instancenorm1(self.conv1(x)))
        x = self.leakyrelu(self.Instancenorm2(self.conv2(x)))
        x = self.leakyrelu(self.Instancenorm3(self.conv3(x)))
        x = self.leakyrelu((self.fc1(x)))
        x = self.sigmoid((self.fc2(x)))               
        return x