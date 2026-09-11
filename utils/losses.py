import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class SSIMLoss(nn.Module):
    """
    Structural Similarity Index Measure (SSIM) Loss.
    Measures perceptual similarity between two images.
    """
    def __init__(self, window_size=11, channel=3):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.channel = channel
        self.window = self.create_window(window_size, channel)

    def gaussian(self, window_size, sigma):
        gauss = torch.exp(torch.tensor([-(x - window_size // 2) ** 2 / float(2 * sigma ** 2) for x in range(window_size)]))
        return gauss / gauss.sum()

    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def forward(self, img1, img2):
        if self.window.device != img1.device:
            self.window = self.window.to(img1.device)

        window = self.window

        mu1 = F.conv2d(img1, window, padding=self.window_size // 2, groups=self.channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size // 2, groups=self.channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=self.window_size // 2, groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=self.window_size // 2, groups=self.channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=self.window_size // 2, groups=self.channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return 1.0 - ssim_map.mean()


class PerceptualVGGFeatures(nn.Module):
    def __init__(self):
        super(PerceptualVGGFeatures, self).__init__()
        try:
            vgg = models.vgg16(pretrained=True).features
            self.slice1 = nn.Sequential(*[vgg[x] for x in range(4)])
            self.slice2 = nn.Sequential(*[vgg[x] for x in range(4, 9)])
            self.slice3 = nn.Sequential(*[vgg[x] for x in range(9, 16)])
            for param in self.parameters():
                param.requires_grad = False
            self.initialized = True
        except Exception:
            self.initialized = False

    def forward(self, x):
        if not self.initialized:
            return []
        h1 = self.slice1(x)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)
        return [h1, h2, h3]


class CompositeLoss(nn.Module):
    """
    Combined loss function: L1 Loss + SSIM Loss + Perceptual VGG Loss.
    Prevents blurry restoration outputs in cloud removal models.
    """
    def __init__(self, lambda_l1=1.0, lambda_ssim=0.5, lambda_perceptual=0.1):
        super(CompositeLoss, self).__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_ssim = lambda_ssim
        self.lambda_perceptual = lambda_perceptual

        self.l1_loss = nn.L1Loss()
        self.ssim_loss = SSIMLoss()
        self.vgg = PerceptualVGGFeatures()

    def forward(self, pred, target):
        loss = self.lambda_l1 * self.l1_loss(pred, target)

        if self.lambda_ssim > 0:
            loss = loss + self.lambda_ssim * self.ssim_loss(pred, target)

        if self.lambda_perceptual > 0 and self.vgg.initialized:
            try:
                pred_vgg = self.vgg(pred)
                target_vgg = self.vgg(target)
                perceptual_loss = 0.0
                for p_feat, t_feat in zip(pred_vgg, target_vgg):
                    perceptual_loss += self.l1_loss(p_feat, t_feat)
                loss = loss + self.lambda_perceptual * perceptual_loss
            except Exception:
                pass

        return loss

if __name__ == "__main__":
    x = torch.rand(2, 3, 256, 256)
    y = torch.rand(2, 3, 256, 256)
    criterion = CompositeLoss()
    val = criterion(x, y)
    print(f"Composite Loss verification value: {val.item():.4f}")
