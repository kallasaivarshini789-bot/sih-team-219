import numpy as np
import cv2
from skimage.exposure import match_histograms

def match_colors(source_img, reference_img):
    """
    Matches the color histogram of source_img to reference_img.
    Prevents color tint/shift issues after deep learning reconstruction.
    """
    try:
        matched = match_histograms(source_img, reference_img, channel_axis=-1)
        return matched.astype(np.uint8)
    except Exception:
        return source_img

def blend_restored_image(original_cloudy, restored_pred, cloud_mask, feather_radius=7):
    """
    Seamlessly blends the AI-reconstructed cloud-free prediction into the non-cloudy regions 
    of the original image using distance transform alpha feathering.
    
    Args:
        original_cloudy (np.ndarray): Original input image uint8 (H, W, 3)
        restored_pred (np.ndarray): AI restored prediction uint8 (H, W, 3)
        cloud_mask (np.ndarray): Binary cloud mask uint8 (H, W) where 255 = cloud
        feather_radius (int): Smoothness of transition boundary
        
    Returns:
        blended (np.ndarray): Seamless composite image uint8 (H, W, 3)
    """
    if original_cloudy.shape != restored_pred.shape:
        restored_pred = cv2.resize(restored_pred, (original_cloudy.shape[1], original_cloudy.shape[0]))
        
    if cloud_mask.shape[:2] != original_cloudy.shape[:2]:
        cloud_mask = cv2.resize(cloud_mask, (original_cloudy.shape[1], original_cloudy.shape[0]))

    # Match color statistics of restored prediction to original clear regions
    matched_pred = match_colors(restored_pred, original_cloudy)

    # Blur the mask to create soft alpha blending boundary
    alpha_mask = cv2.GaussianBlur(cloud_mask.astype(np.float32) / 255.0, (feather_radius*2+1, feather_radius*2+1), 0)
    alpha_mask = np.expand_dims(alpha_mask, axis=-1)  # (H, W, 1)

    # Composite: non-cloudy regions come directly from original image; cloudy regions come from model
    blended_float = (1.0 - alpha_mask) * original_cloudy.astype(np.float32) + alpha_mask * matched_pred.astype(np.float32)
    blended = np.clip(blended_float, 0, 255).astype(np.uint8)

    return blended

if __name__ == "__main__":
    orig = np.full((256, 256, 3), 100, dtype=np.uint8)
    pred = np.full((256, 256, 3), 150, dtype=np.uint8)
    mask = np.zeros((256, 256), dtype=np.uint8)
    mask[50:150, 50:150] = 255
    res = blend_restored_image(orig, pred, mask)
    print("Blending verification shape:", res.shape)
