import torch
import torch.nn as nn


class MinibatchDiscrimination(nn.Module):
    def __init__(self, in_features, out_features, kernel_dims, mean=True):
        super(MinibatchDiscrimination, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_dims = kernel_dims
        self.mean = mean
        self.T = nn.Parameter(torch.Tensor(in_features, out_features, kernel_dims))
        torch.nn.init.normal_(self.T, 0, 1)
    def forward(self, x):
        matrices = x.mm(self.T.view(self.in_features, -1))
        matrices = matrices.view(-1, self.out_features, self.kernel_dims)
        M = matrices.unsqueeze(0)
        M_T = M.permute(1, 0, 2, 3)
        norm = torch.abs(M - M_T).sum(3)
        expnorm = torch.exp(-norm)
        o_b = expnorm.sum(0) - 1
        if self.mean:
            o_b /= x.size(0) - 1
        x = torch.cat([x, o_b], 1)
        return x
    
class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()                
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        self.leaky_relu1 = nn.LeakyReLU(0.2, inplace=True)
        self.avgpool1 = nn.AvgPool2d(2, stride=2)
        self.dropout1 = nn.Dropout2d(0.2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        self.leaky_relu2 = nn.LeakyReLU(0.2, inplace=True)
        self.avgpool2 = nn.AvgPool2d(2, stride=2)
        self.dropout2 = nn.Dropout2d(0.2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        self.leaky_relu3 = nn.LeakyReLU(0.2, inplace=True)
        self.avgpool3 = nn.AvgPool2d(2, stride=2)
        self.dropout3 = nn.Dropout2d(0.2)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        self.leaky_relu4 = nn.LeakyReLU(0.2, inplace=True)
        self.avgpool4 = nn.AvgPool2d(2, stride=2)
        self.dropout4 = nn.Dropout2d(0.2)
        self.minibatch_discrimination = MinibatchDiscrimination(128, 32, 32)
        self.fc = nn.Linear(160, 1)

    def forward(self, img):
        x = self.conv1(img)
        x = self.leaky_relu1(x)
        x = self.avgpool1(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.leaky_relu2(x)
        x = self.avgpool2(x)
        x = self.dropout2(x)
        x = self.conv3(x)
        x = self.leaky_relu3(x)
        x = self.avgpool3(x)
        x = self.dropout3(x)
        x = self.conv4(x)
        x = self.leaky_relu4(x)
        x = self.avgpool4(x)
        x = self.dropout4(x)
        x = x.view(x.shape[0], -1)
        x = self.minibatch_discrimination(x)
        x = self.fc(x)        
        return x
    


