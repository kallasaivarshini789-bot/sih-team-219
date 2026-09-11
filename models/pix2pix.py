import torch
import torch.nn as nn

try:
    from models.unet import UNet
except ModuleNotFoundError:
    from unet import UNet

class Pix2PixGenerator(nn.Module):
    """
    Generator network for Pix2Pix Cloud Removal GAN.
    Uses U-Net backbone with residual skip connections.
    """
    def __init__(self, in_channels=3, out_channels=3):
        super(Pix2PixGenerator, self).__init__()
        self.unet = UNet(n_channels=in_channels, n_classes=out_channels)

    def forward(self, x):
        return self.unet(x)


class PatchGANDiscriminator(nn.Module):
    """
    70x70 PatchGAN Discriminator for Pix2Pix GAN.
    Evaluates local patch realism rather than whole-image classification,
    forcing sharp high-frequency texture reconstruction in cloud removal.
    """
    def __init__(self, in_channels=6, ndf=64):
        super(PatchGANDiscriminator, self).__init__()

        self.model = nn.Sequential(
            # input is (in_channels) x 256 x 256
            nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # 1-channel prediction map
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=1)
        )

    def forward(self, cloudy_img, target_img):
        # Concatenate cloudy input and target output along channels
        x = torch.cat([cloudy_img, target_img], dim=1)
        return self.model(x)

if __name__ == "__main__":
    gen = Pix2PixGenerator()
    disc = PatchGANDiscriminator()
    c = torch.randn(1, 3, 256, 256)
    f = torch.randn(1, 3, 256, 256)
    out_g = gen(c)
    out_d = disc(c, out_g)
    print(f"Pix2Pix Generator output: {out_g.shape}, Discriminator output: {out_d.shape}")
