import torch.nn as nn

# Encoder Block
class EncoderBlockV2(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=False, polling=True, bn=False):
        super(EncoderBlockV2, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.InstanceNorm2d(out_channels) if bn else nn.GroupNorm(32, out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.InstanceNorm2d(out_channels) if bn else nn.GroupNorm(32, out_channels)
        self.relu2 = nn.ReLU(inplace=True)

        self.dropout = nn.Dropout() if dropout else None
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) if polling else None

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu2(x)

        if self.dropout:
            x = self.dropout(x)

        if self.pool:
            x = self.pool(x)

        return x
