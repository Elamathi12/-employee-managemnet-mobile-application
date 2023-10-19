import torch.nn as nn
import torch.nn.functional as F

class UNetV2(nn.Module):
    def __init__(self, num_classes, in_channels=3, bn=False):
        super(UNetV2, self).__init__()

        # Encoder Blocks
        self.enc1_conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.enc1_norm1 = nn.InstanceNorm2d(64) if bn else nn.GroupNorm(32, 64)
        self.enc1_relu1 = nn.ReLU(inplace=True)
        self.enc1_conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.enc1_norm2 = nn.InstanceNorm2d(64) if bn else nn.GroupNorm(32, 64)
        self.enc1_relu2 = nn.ReLU(inplace=True)

        self.enc2_conv1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.enc2_norm1 = nn.InstanceNorm2d(128) if bn else nn.GroupNorm(32, 128)
        self.enc2_relu1 = nn.ReLU(inplace=True)
        self.enc2_conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.enc2_norm2 = nn.InstanceNorm2d(128) if bn else nn.GroupNorm(32, 128)
        self.enc2_relu2 = nn.ReLU(inplace=True)

        self.enc3_conv1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.enc3_norm1 = nn.InstanceNorm2d(256) if bn else nn.GroupNorm(32, 256)
        self.enc3_relu1 = nn.ReLU(inplace=True)
        self.enc3_conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.enc3_norm2 = nn.InstanceNorm2d(256) if bn else nn.GroupNorm(32, 256)
        self.enc3_relu2 = nn.ReLU(inplace=True)

        self.enc4_conv1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.enc4_norm1 = nn.InstanceNorm2d(512) if bn else nn.GroupNorm(32, 512)
        self.enc4_relu1 = nn.ReLU(inplace=True)
        self.enc4_conv2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.enc4_norm2 = nn.InstanceNorm2d(512) if bn else nn.GroupNorm(32, 512)
        self.enc4_relu2 = nn.ReLU(inplace=True)

        self.pooling = nn.AvgPool2d(kernel_size=2, stride=2)

        # Center Block
        self.center_conv1 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.center_norm1 = nn.InstanceNorm2d(1024) if bn else nn.GroupNorm(32, 1024)
        self.center_relu1 = nn.ReLU(inplace=True)
        self.center_conv2 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.center_norm2 = nn.InstanceNorm2d(512) if bn else nn.GroupNorm(32, 512)
        self.center_relu2 = nn.ReLU(inplace=True)
        self.center_conv_transpose = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)

        # Decoder Blocks
        self.dec4_conv_transpose = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec3_conv_transpose = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec2_conv_transpose = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)

        self.final_conv1 = nn.Conv2d(512, num_classes, kernel_size=1)
        self.final_conv2 = nn.Conv2d(256, num_classes, kernel_size=1)
        self.final_conv3 = nn.Conv2d(128, num_classes, kernel_size=1)
        self.final_conv4 = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        enc1 = self.enc1_conv1(x)
        enc1 = self.enc1_norm1(enc1)
        enc1 = self.enc1_relu1(enc1)
        enc1 = self.enc1_conv2(enc1)
        enc1 = self.enc1_norm2(enc1)
        enc1 = self.enc1_relu2(enc1)

        enc2 = self.enc2_conv1(enc1)
        enc2 = self.enc2_norm1(enc2)
        enc2 = self.enc2_relu1(enc2)
        enc2 = self.enc2_conv2(enc2)
        enc2 = self.enc2_norm2(enc2)
        enc2 = self.enc2_relu2(enc2)

        enc3 = self.enc3_conv1(enc2)
        enc3 = self.enc3_norm1(enc3)
        enc3 = self.enc3_relu1(enc3)
        enc3 = self.enc3_conv2(enc3)
        enc3 = self.enc3_norm2(enc3)
        enc3 = self.enc3_relu2(enc3)

        enc4 = self.enc4_conv1(enc3)
        enc4 = self.enc4_norm1(enc4)
        enc4 = self.enc4_relu1(enc4)
        enc4 = self.enc4_conv2(enc4)
        enc4 = self.enc4_norm2(enc4)
        enc4 = self.enc4_relu2(enc4)

        center = self.center_conv1(enc4)
        center = self.center_norm1(center)
        center = self.center_relu1(center)
        center = self.center_conv2(center)
        center = self.center_norm2(center)
        center = self.center_relu2
