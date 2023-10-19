import torch.nn as nn

class UNetVGG(nn.Module):
    def __init__(self, out_channels=1, in_channels=1, bn=False):
        super(UNetVGG, self).__init__()

        # Encoder Blocks
        self.enc1_conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.enc1_norm1 = nn.BatchNorm2d(64) if bn else nn.GroupNorm(32, 64)
        self.enc1_relu1 = nn.ReLU(inplace=True)
        self.enc1_conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.enc1_norm2 = nn.BatchNorm2d(64) if bn else nn.GroupNorm(32, 64)
        self.enc1_relu2 = nn.ReLU(inplace=True)

        self.enc2_conv1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.enc2_norm1 = nn.BatchNorm2d(128) if bn else nn.GroupNorm(32, 128)
        self.enc2_relu1 = nn.ReLU(inplace=True)
        self.enc2_conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.enc2_norm2 = nn.BatchNorm2d(128) if bn else nn.GroupNorm(32, 128)
        self.enc2_relu2 = nn.ReLU(inplace=True)

        self.enc3_conv1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.enc3_norm1 = nn.BatchNorm2d(256) if bn else nn.GroupNorm(32, 256)
        self.enc3_relu1 = nn.ReLU(inplace=True)
        self.enc3_conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.enc3_norm2 = nn.BatchNorm2d(256) if bn else nn.GroupNorm(32, 256)
        self.enc3_relu2 = nn.ReLU(inplace=True)

        self.enc4_conv1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.enc4_norm1 = nn.BatchNorm2d(512) if bn else nn.GroupNorm(32, 512)
        self.enc4_relu1 = nn.ReLU(inplace=True)
        self.enc4_conv2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.enc4_norm2 = nn.BatchNorm2d(512) if bn else nn.GroupNorm(32, 512)
        self.enc4_relu2 = nn.ReLU(inplace=True)

        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)

        # Center Block
        self.center_conv1 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.center_norm1 = nn.BatchNorm2d(1024) if bn else nn.GroupNorm(32, 1024)
        self.center_relu1 = nn.ReLU(inplace=True)
        self.center_conv2 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.center_norm2 = nn.BatchNorm2d(512) if bn else nn.GroupNorm(32, 512)
        self.center_relu2 = nn.ReLU(inplace=True)
        self.center_conv_transpose = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)

        # Decoder Blocks
        self.dec4_conv_transpose = nn.ConvTranspose2d(256, 512, kernel_size=2, stride=2)
        self.dec3_conv_transpose = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec2_conv_transpose = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)

        self.final_conv = nn.Conv2d(128, out_channels, kernel_size=1)

    def forward(self, x):
        print(x.shape,"input tensor shape")
        enc1 = self.enc1_conv1(x)
        print(enc1.shape,"1")
        enc1 = self.enc1_norm1(enc1)
        print(enc1.shape,"2")
        enc1 = self.enc1_relu1(enc1)
        print(enc1 .shape,"3")
        enc1 = self.enc1_conv2(enc1)
        print(enc1 .shape,"4")
        enc1 = self.enc1_norm2(enc1)
        print(enc1 .shape,"5")
        enc1 = self.enc1_relu2(enc1)
        print(enc1 .shape,"6")

        enc2 = self.enc2_conv1(enc1)
        print( enc2.shape,"7")
        enc2 = self.enc2_norm1(enc2)
        print( enc2.shape,"8")
        enc2 = self.enc2_relu1(enc2)
        print( enc2.shape,"9")
        enc2 = self.enc2_conv2(enc2)
        print( enc2.shape,"10")
        enc2 = self.enc2_norm2(enc2)
        print( enc2.shape,"11")
        enc2 = self.enc2_relu2(enc2)
        print( enc2.shape,"12")


        enc3 = self.enc3_conv1(enc2)
        print(enc3.shape,"13")
        enc3 = self.enc3_norm1(enc3)
        print(enc3.shape,"14")
        enc3 = self.enc3_relu1(enc3)
        print(enc3.shape,"15")
        enc3 = self.enc3_conv2(enc3)
        print(enc3.shape,"16")
        enc3 = self.enc3_norm2(enc3)
        print(enc3.shape,"17")
        enc3 = self.enc3_relu2(enc3)
        print(enc3.shape,"18")


        enc4 = self.enc4_conv1(enc3)
        print( enc4.shape,"19")
        enc4 = self.enc4_norm1(enc4)
        print( enc4.shape,"20")
        enc4 = self.enc4_relu1(enc4)
        print( enc4.shape,"21")
        enc4 = self.enc4_conv2(enc4)
        print( enc4.shape,"22")
        enc4 = self.enc4_norm2(enc4)
        print( enc4.shape,"23")
        enc4 = self.enc4_relu2(enc4)
        print( enc4.shape,"24")

        center = self.center_conv1(enc4)
        print(center.shape,"25")
        center = self.center_norm1(center)
        print(center.shape,"26")
        center = self.center_relu1(center)
        print(center.shape,"27")
        center = self.center_conv2(center)
        print(center.shape,"28")
        center = self.center_norm2(center)
        print(center.shape,"29")
        center = self.center_relu2(center)
        print(center.shape,"30")
        center = self.center_conv_transpose(center)
        print(center.shape,"31")


        dec4 = self.dec4_conv_transpose(center)
        print(dec4.shape,"32")
        dec3 = self.dec3_conv_transpose(dec4)
        print(dec3.shape,"33")
        dec2 = self.dec2_conv_transpose(dec3)
        print(dec2 .shape,"34")
        final = self.final_conv(dec2)
        print(final.shape,"35")
        return final
