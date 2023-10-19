import torch.nn as nn

class DecoderBlockV2(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, bn=False):
        super(DecoderBlockV2, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1)
        self.norm1 = nn.InstanceNorm2d(middle_channels) if bn else nn.GroupNorm(32, middle_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.InstanceNorm2d(out_channels) if bn else nn.GroupNorm(32, out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu2(x)

        return x
