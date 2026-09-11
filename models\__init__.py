from models.unet import UNet
from models.attention_unet import AttentionUNet
from models.pix2pix import Pix2PixGenerator, PatchGANDiscriminator

def get_model(model_name="unet", in_channels=3, out_channels=3):
    """
    Factory function to retrieve model instance by name.
    
    Supported model names:
      - 'unet': Standard Lightweight U-Net
      - 'attention_unet': Attention U-Net with Spatial Attention Gates
      - 'pix2pix': Pix2Pix Conditional GAN Generator
    """
    model_name = model_name.lower().strip()
    if model_name == "unet":
        return UNet(n_channels=in_channels, n_classes=out_channels)
    elif model_name in ["attention_unet", "attention-unet", "att_unet"]:
        return AttentionUNet(n_channels=in_channels, n_classes=out_channels)
    elif model_name == "pix2pix":
        return Pix2PixGenerator(in_channels=in_channels, out_channels=out_channels)
    else:
        raise ValueError(f"Unknown model name '{model_name}'. Choose from ['unet', 'attention_unet', 'pix2pix'].")
