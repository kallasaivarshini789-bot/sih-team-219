import numpy as np
import cv2
from PIL import Image

def get_preset_samples():
    """
    Generates realistic high-quality synthetic satellite image presets for the web dashboard.
    Presets include:
      1. Amazon Basin Rainforest (Dense Cumulus)
      2. Coastal Harbor & Urban (Thin Cirrus)
      3. Agricultural Valley (Layered Overcast)
      4. Alpine Snow Peak (Hazy Stratus)
    """
    presets = {}
    size = (384, 384)
    h, w = size

    # 1. Amazon Rainforest Preset
    rainforest = np.zeros((h, w, 3), dtype=np.uint8)
    rainforest[:, :, 0] = np.random.randint(15, 45, size, dtype=np.uint8)   # Red
    rainforest[:, :, 1] = np.random.randint(90, 160, size, dtype=np.uint8)  # Green
    rainforest[:, :, 2] = np.random.randint(20, 60, size, dtype=np.uint8)   # Blue
    # Add meandering river
    y, x = np.ogrid[:h, :w]
    river_path = np.abs(y - (120 + 40 * np.sin(x / 40.0))) < 12
    rainforest[river_path] = [20, 70, 140]
    
    # Heavy cloud cover
    cloud_r = np.zeros((h, w), dtype=np.float32)
    cv2.circle(cloud_r, (140, 140), 90, 1.0, -1)
    cv2.circle(cloud_r, (220, 180), 75, 0.95, -1)
    cv2.circle(cloud_r, (100, 240), 65, 0.85, -1)
    cloud_r = cv2.GaussianBlur(cloud_r, (45, 45), 0)
    cloud_r_3d = np.expand_dims(cloud_r, axis=-1)
    cloudy_rainforest = ((1.0 - cloud_r_3d) * rainforest + cloud_r_3d * 245).clip(0, 255).astype(np.uint8)

    presets["Amazon Rainforest (Dense Cumulus)"] = {
        "cloudy": Image.fromarray(cloudy_rainforest),
        "cloud_free": Image.fromarray(rainforest),
        "desc": "Dense tropical rainforest with heavy cumulus cloud cover over river channel."
    }

    # 2. Coastal Harbor & Urban Preset
    coastal = np.zeros((h, w, 3), dtype=np.uint8)
    # Ocean water
    coastal[:, :w//2] = [25, 85, 170]
    # Urban land
    coastal[:, w//2:] = [140, 135, 125]
    # City grid lines
    coastal[::30, w//2:, :] = [180, 175, 165]
    coastal[:, w//2::30, :] = [180, 175, 165]
    
    # Thin cirrus wisps
    cirrus = np.zeros((h, w), dtype=np.float32)
    cv2.ellipse(cirrus, (200, 150), (160, 40), 30, 0, 360, 0.6, -1)
    cv2.ellipse(cirrus, (150, 260), (140, 30), -20, 0, 360, 0.5, -1)
    cirrus = cv2.GaussianBlur(cirrus, (31, 31), 0)
    cirrus_3d = np.expand_dims(cirrus, axis=-1)
    cloudy_coastal = ((1.0 - cirrus_3d) * coastal + cirrus_3d * 240).clip(0, 255).astype(np.uint8)

    presets["Coastal Harbor & Urban (Thin Cirrus)"] = {
        "cloudy": Image.fromarray(cloudy_coastal),
        "cloud_free": Image.fromarray(coastal),
        "desc": "Coastal city port featuring thin cirrus clouds over ocean and urban grid."
    }

    # 3. Agricultural Fields Preset
    agri = np.zeros((h, w, 3), dtype=np.uint8)
    # Crop patches
    for row in range(0, h, 48):
        for col in range(0, w, 48):
            color = [np.random.randint(40, 90), np.random.randint(110, 200), np.random.randint(30, 80)]
            agri[row:row+48, col:col+48] = color
            
    # Cloud layer
    cloud_ag = np.zeros((h, w), dtype=np.float32)
    cv2.rectangle(cloud_ag, (60, 60), (320, 280), 0.75, -1)
    cloud_ag = cv2.GaussianBlur(cloud_ag, (55, 55), 0)
    cloud_ag_3d = np.expand_dims(cloud_ag, axis=-1)
    cloudy_agri = ((1.0 - cloud_ag_3d) * agri + cloud_ag_3d * 250).clip(0, 255).astype(np.uint8)

    presets["Agricultural Fields (Overcast)"] = {
        "cloudy": Image.fromarray(cloudy_agri),
        "cloud_free": Image.fromarray(agri),
        "desc": "Patchwork farmlands obscured by mid-altitude overcast cloud layer."
    }

    return presets

if __name__ == "__main__":
    p = get_preset_samples()
    print(f"Generated {len(p)} preset sample scenes.")
