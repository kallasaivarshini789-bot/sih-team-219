import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from models.unet import DoubleConv, Down, OutConv
except ModuleNotFoundError:
    from unet import DoubleConv, Down, OutConv

class AttentionGate(nn.Module):
    """
    Attention Gate (AG) mechanism for U-Net.
    Allows the model to focus on cloud features while suppressing uninformative background signals.
    """
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_l = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_l(x)
        
        # Match dimensions if needed
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=True)
            
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        return x * psi


class AttentionUp(nn.Module):
    """Upsampling block with Attention Gate filtering"""
    def __init__(self, in_channels, out_channels, skip_channels, bilinear=True):
        super(AttentionUp, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            g_channels = in_channels // 2
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
            g_channels = in_channels // 2
            
        self.ag = AttentionGate(F_g=g_channels, F_l=skip_channels, F_int=out_channels)
        self.conv = DoubleConv(g_channels + skip_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Pad if sizes differ
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        
        # Apply Attention Gate on skip connection x2 using upsampled feature map x1
        x2_att = self.ag(g=x1, x=x2)
        
        x = torch.cat([x2_att, x1], dim=1)
        return self.conv(x)


class AttentionUNet(nn.Module):
    """
    Attention U-Net Architecture for Satellite Cloud Removal.
    Integrates Attention Gates into decoder skip connections.
    """
    def __init__(self, n_channels=3, n_classes=3, bilinear=True):
        super(AttentionUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        
        factor = 2 if bilinear else 1
        self.down4 = Down(256, 512 // factor) # 256 channels
        
        self.up1 = AttentionUp(in_channels=512, out_channels=128, skip_channels=256, bilinear=bilinear)
        self.up2 = AttentionUp(in_channels=256, out_channels=64, skip_channels=128, bilinear=bilinear)
        self.up3 = AttentionUp(in_channels=128, out_channels=32, skip_channels=64, bilinear=bilinear)
        self.up4 = AttentionUp(in_channels=64, out_channels=32, skip_channels=32, bilinear=bilinear)
        
        self.outc = OutConv(32, n_classes)

    def forward(self, x):
        x1 = self.inc(x)        # 32
        x2 = self.down1(x1)     # 64
        x3 = self.down2(x2)     # 128
        x4 = self.down3(x3)     # 256
        x5 = self.down4(x4)     # 256
        
        x = self.up1(x5, x4)    # out 128
        x = self.up2(x, x3)     # out 64
        x = self.up3(x, x2)     # out 32
        x = self.up4(x, x1)     # out 32
        return self.outc(x)

if __name__ == "__main__":
    model = AttentionUNet(3, 3)
    inp = torch.randn(1, 3, 256, 256)
    out = model(inp)
    print(f"Attention U-Net test -> input: {inp.shape} -> output: {out.shape}")
