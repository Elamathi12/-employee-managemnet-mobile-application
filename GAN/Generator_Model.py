import torch 
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self,input_shape): # INPUT : 20 X 1 X 100
        super(Generator, self).__init__()

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ TRANSPOSE CONVOLUTIONAL LAYERS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

        # INPUT : 100 X 1 X 1000
        self.convtranspose1 = nn.ConvTranspose1d(in_channels=1 ,out_channels=8,kernel_size=6, stride=4, padding=1)
        self.Instancenorm1 = nn.BatchNorm1d(8)
        # OUTPUT : 100 X 8 X 4000
        
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ CONVOLUTIONAL LAYERS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

        # INPUT : 100 X 8 X 4000
        self.conv1 = nn.Conv1d(in_channels=8 ,out_channels=1,kernel_size=4, stride=2, padding=1)
        self.Instancenorm2 = nn.BatchNorm1d(1)

        # OUTPUT : 100 X 1 X 2000

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MAXPOOL LAYERS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

        # INPUT : 100 X 1 X 2000
        self.maxpool = nn.MaxPool1d(kernel_size=4, stride=2, padding=1)
        self.Instancenorm3 = nn.BatchNorm1d(1)

        # OUTPUT : 100 X 1 X 1000

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ACTIVATION LAYER ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()


    def forward(self,x):
        print("~~~~~~~~~Entered forward function in generator model~~~~~~~~~~~~") 
        x = self.leakyrelu(self.Instancenorm1(self.convtranspose1(x)))
        x = self.leakyrelu(self.Instancenorm2(self.conv1(x)))
        x = self.leakyrelu(self.Instancenorm3(self.maxpool(x)))
        x = self.sigmoid(x)               
        return x