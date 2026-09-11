import torch
import numpy as np
from skimage.metrics import structural_similarity as calculate_ssim
from skimage.metrics import peak_signal_noise_ratio as calculate_psnr

def compute_metrics(img_pred, img_target):
    """
    Computes image quality metrics between predicted cloud-free image and target ground truth image.
    
    Args:
        img_pred (np.ndarray or torch.Tensor): Restored image (H, W, C) in range [0, 255] or [0, 1]
        img_target (np.ndarray or torch.Tensor): Ground truth image (H, W, C) in same range
        
    Returns:
        dict: {'PSNR': float, 'SSIM': float, 'MAE': float}
    """
    if isinstance(img_pred, torch.Tensor):
        img_pred = img_pred.detach().cpu().numpy()
    if isinstance(img_target, torch.Tensor):
        img_target = img_target.detach().cpu().numpy()
        
    # Ensure float32 in range [0, 1]
    if img_pred.max() > 1.0:
        img_pred = img_pred.astype(np.float32) / 255.0
    if img_target.max() > 1.0:
        img_target = img_target.astype(np.float32) / 255.0

    # Ensure shape (H, W, C)
    if img_pred.shape[0] in [1, 3] and img_pred.ndim == 3:
        img_pred = np.transpose(img_pred, (1, 2, 0))
    if img_target.shape[0] in [1, 3] and img_target.ndim == 3:
        img_target = np.transpose(img_target, (1, 2, 0))

    mae = np.mean(np.abs(img_pred - img_target))
    
    try:
        psnr = calculate_psnr(img_target, img_pred, data_range=1.0)
    except Exception:
        psnr = 0.0
        
    try:
        ssim = calculate_ssim(img_target, img_pred, channel_axis=-1, data_range=1.0)
    except Exception:
        # Fallback if skimage version differs
        try:
            ssim = calculate_ssim(img_target, img_pred, multichannel=True, data_range=1.0)
        except Exception:
            ssim = 0.0
            
    return {
        "PSNR": round(float(psnr), 2),
        "SSIM": round(float(ssim), 4),
        "MAE": round(float(mae), 4)
    }

if __name__ == "__main__":
    a = np.random.rand(256, 256, 3).astype(np.float32)
    b = a + np.random.normal(0, 0.05, a.shape).astype(np.float32)
    b = np.clip(b, 0.0, 1.0)
    metrics = compute_metrics(b, a)
    print("Metrics verification output:", metrics)
