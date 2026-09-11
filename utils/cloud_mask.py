import numpy as np
import cv2
from PIL import Image

def detect_clouds(image_np, brightness_threshold=180, saturation_threshold=50):
    """
    Detects cloud cover in an optical satellite RGB image.
    Clouds typically exhibit high brightness (luminance) and low saturation (whiteness/grayness).
    
    Args:
        image_np (np.ndarray): Image array of shape (H, W, 3) with values in range [0, 255]
        brightness_threshold (int): Minimum intensity (0-255) to qualify as bright
        saturation_threshold (int): Maximum saturation (0-255) in HSV space to qualify as white/grey
        
    Returns:
        mask (np.ndarray): Binary mask (H, W) where 255 = cloud, 0 = clear
        cloud_pct (float): Percentage of cloud coverage (0.0 to 100.0%)
        heatmap (np.ndarray): Color-mapped overlay image (H, W, 3) displaying cloud confidence
    """
    if image_np.dtype != np.uint8:
        image_np = (image_np * 255.0).clip(0, 255).astype(np.uint8)

    # Convert to HSV color space
    hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
    _, s, v = cv2.split(hsv)

    # Luminance channel
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    # Cloud criteria: High Value/Luminance AND Low Saturation
    cloud_mask = (v >= brightness_threshold) & (gray >= brightness_threshold) & (s <= saturation_threshold)
    mask_binary = (cloud_mask * 255).astype(np.uint8)

    # Morphological cleaning to close small holes and smooth edges
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_cleaned = cv2.morphologyEx(mask_binary, cv2.MORPH_CLOSE, kernel)
    mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_OPEN, kernel)

    # Calculate cloud coverage percentage
    cloud_pct = float(np.sum(mask_cleaned > 0)) / float(mask_cleaned.size) * 100.0

    # Create heatmap overlay (Red for detected clouds)
    heatmap = cv2.applyColorMap(mask_cleaned, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Blend with original image
    overlay = cv2.addWeighted(image_np, 0.6, heatmap, 0.4, 0)

    return mask_cleaned, round(cloud_pct, 2), overlay

if __name__ == "__main__":
    dummy_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    dummy_img[50:150, 50:150] = 240  # Synthetic bright patch
    mask, pct, overlay = detect_clouds(dummy_img)
    print(f"Cloud detection test -> Cloud coverage: {pct}%")
