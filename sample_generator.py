"""
Sample Satellite Scene Generator
--------------------------------
Generates ultra-realistic synthetic satellite scenes with simulated cloud cover
(Cumulus clouds, thin cirrus haze, and optical distortions) for instant 1-click testing.
"""

import os
import numpy as np
import cv2
from PIL import Image

def generate_perlin_noise(width, height, scale=50.0, octaves=4):
    """Generate multi-octave smooth perlin-like noise texture."""
    noise = np.zeros((height, width), dtype=np.float32)
    for i in range(octaves):
        freq = 2 ** i
        amp = 1.0 / freq
        # Low frequency random grid
        gw = max(int(width / (scale / freq)), 4)
        gh = max(int(height / (scale / freq)), 4)
        grid = np.random.rand(gh, gw).astype(np.float32)
        resized = cv2.resize(grid, (width, height), interpolation=cv2.INTER_CUBIC)
        noise += amp * resized
    noise = (noise - np.min(noise)) / (np.max(noise) - np.min(noise) + 1e-6)
    return noise

def create_agriculture_scene(w=640, h=480):
    """Generates an agricultural landscape with crop parcels and river.
    Tuned so moderate/crop vegetation (yellow NDVI) is the dominant class.
    """
    img = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Base terrain — mostly mature/moderate crop fields in pixel color
    # RGB tuned so Green is slightly > Red (moderate VARI 0.02-0.18 = YELLOW zone)
    grid_size = 45
    for y in range(0, h, grid_size):
        for x in range(0, w, grid_size):
            # 30% dense crop, 45% mature crop, 15% harvested/dry, 10% bare soil
            rng = np.random.rand()
            if rng < 0.30:
                # Dense healthy (GREEN NDVI) — BGR: high G, low R, low B
                color = [np.random.randint(18, 40),   # B
                         np.random.randint(148, 195),  # G  (much higher than R)
                         np.random.randint(28, 58)]    # R
            elif rng < 0.75:
                # Moderate/mature crop (YELLOW NDVI) — BGR: G slightly > R
                color = [np.random.randint(30, 65),   # B
                         np.random.randint(155, 205),  # G  (slightly > R)
                         np.random.randint(120, 165)]  # R
            elif rng < 0.90:
                # Harvested / yellowing (YELLOW→RED border) — BGR: G ≈ R
                color = [np.random.randint(50, 80),   # B
                         np.random.randint(140, 175),  # G  (≈ R)
                         np.random.randint(130, 170)]  # R
            else:
                # Bare soil (RED NDVI) — BGR: R > G
                color = [np.random.randint(40, 70),   # B
                         np.random.randint(90, 130),   # G  (< R)
                         np.random.randint(145, 190)]  # R
            
            cv2.rectangle(img, (x, y), (x + grid_size, y + grid_size), color, -1)
            # Thin field boundary lines
            cv2.rectangle(img, (x, y), (x + grid_size, y + grid_size), (25, 45, 25), 1)

    # Winding river (blue water)
    curve_pts = []
    for y_step in range(0, h + 20, 20):
        x_pos = int(w * 0.38 + 35 * np.sin(y_step * 0.022) + 18 * np.cos(y_step * 0.055))
        curve_pts.append((x_pos, y_step))
    curve_pts = np.array(curve_pts, dtype=np.int32)
    cv2.polylines(img, [curve_pts], False, (190, 100, 25), 14)  # River BGR

    # Lighter cloud/haze (reduced alpha so more terrain shows through)
    cloud_noise = generate_perlin_noise(w, h, scale=90.0, octaves=4)
    cloud_mask = np.clip((cloud_noise - 0.48) * 1.6, 0.0, 0.65)  # Lighter clouds

    cloud_layer = np.full((h, w, 3), 245, dtype=np.uint8)
    for c in range(3):
        img[:, :, c] = (img[:, :, c] * (1.0 - cloud_mask) + cloud_layer[:, :, c] * cloud_mask).astype(np.uint8)

    # Mild haze only
    img = cv2.addWeighted(img, 0.88, np.full_like(img, 225), 0.12, 0)
    return img

def create_coastal_scene(w=640, h=480):
    """Generates a coastal delta scene with deep ocean, beaches, and estuary."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Deep Ocean (Left/Bottom)
    for y in range(h):
        for x in range(w):
            coast_dist = x - (w * 0.45 + 50 * np.sin(y * 0.015))
            if coast_dist < -20:
                # Deep Ocean
                img[y, x] = [np.random.randint(160, 210), np.random.randint(70, 110), np.random.randint(15, 40)] # BGR Deep Blue
            elif coast_dist < 0:
                # Shallow Coastal water
                img[y, x] = [np.random.randint(200, 235), np.random.randint(140, 180), np.random.randint(30, 60)] # Cyan Shallow
            elif coast_dist < 15:
                # Sandy beach
                img[y, x] = [np.random.randint(110, 140), np.random.randint(180, 215), np.random.randint(210, 240)] # Sand
            else:
                # Mainland vegetation & delta
                img[y, x] = [np.random.randint(30, 60), np.random.randint(120, 175), np.random.randint(40, 75)]

    # Add coastal sea fog & puffy cumulus clouds
    cloud_noise = generate_perlin_noise(w, h, scale=80.0, octaves=5)
    cloud_mask = np.clip((cloud_noise - 0.4) * 2.0, 0.0, 0.88)
    
    cloud_layer = np.full((h, w, 3), 250, dtype=np.uint8)
    for c in range(3):
        img[:, :, c] = (img[:, :, c] * (1.0 - cloud_mask) + cloud_layer[:, :, c] * cloud_mask).astype(np.uint8)

    img = cv2.addWeighted(img, 0.85, np.full_like(img, 230), 0.15, 0)
    return img

def create_urban_scene(w=640, h=480):
    """Generates an urban metropolitan grid with high building density and roads."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = [90, 95, 100] # Concrete base

    # Road network grid
    road_spacing = 60
    for x in range(0, w, road_spacing):
        cv2.line(img, (x, 0), (x, h), (50, 50, 50), 6) # Main avenues
    for y in range(0, h, road_spacing):
        cv2.line(img, (0, y), (w, y), (50, 50, 50), 6)

    # Buildings inside blocks
    for bx in range(10, w - 40, road_spacing):
        for by in range(10, h - 40, road_spacing):
            # 2 to 4 buildings per block
            for sx in range(bx + 4, bx + road_spacing - 12, 18):
                for sy in range(by + 4, by + road_spacing - 12, 18):
                    roof_color = [np.random.randint(120, 190), np.random.randint(110, 170), np.random.randint(130, 200)]
                    cv2.rectangle(img, (sx, sy), (sx + 14, sy + 14), roof_color, -1)
                    cv2.rectangle(img, (sx, sy), (sx + 14, sy + 14), (40, 40, 40), 1)

    # Add green urban park / lake
    cv2.rectangle(img, (int(w*0.55), int(h*0.45)), (int(w*0.8), int(h*0.75)), (40, 140, 45), -1)
    cv2.circle(img, (int(w*0.68), int(h*0.6)), 28, (190, 95, 20), -1)

    # City smog & thin cloud streaks
    cloud_noise = generate_perlin_noise(w, h, scale=65.0, octaves=4)
    cloud_mask = np.clip((cloud_noise - 0.38) * 1.75, 0.0, 0.82)
    
    cloud_layer = np.full((h, w, 3), 245, dtype=np.uint8)
    for c in range(3):
        img[:, :, c] = (img[:, :, c] * (1.0 - cloud_mask) + cloud_layer[:, :, c] * cloud_mask).astype(np.uint8)

    img = cv2.addWeighted(img, 0.80, np.full_like(img, 220), 0.20, 0)
    return img

def create_mountain_scene(w=640, h=480):
    """Generates high-altitude rugged terrain with dense cloud cover."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    
    elevation = generate_perlin_noise(w, h, scale=120.0, octaves=5)
    for y in range(h):
        for x in range(w):
            elev = elevation[y, x]
            if elev > 0.75:
                # Snow capped ridge
                img[y, x] = [np.random.randint(230, 255), np.random.randint(235, 255), np.random.randint(240, 255)]
            elif elev > 0.5:
                # Rocky crags
                img[y, x] = [np.random.randint(80, 110), np.random.randint(95, 125), np.random.randint(110, 140)]
            else:
                # Dense Pine / Montane forest
                img[y, x] = [np.random.randint(25, 45), np.random.randint(90, 140), np.random.randint(30, 55)]

    # Heavy orographic mountain clouds
    cloud_noise = generate_perlin_noise(w, h, scale=85.0, octaves=4)
    cloud_mask = np.clip((cloud_noise - 0.32) * 2.1, 0.0, 0.90)
    
    cloud_layer = np.full((h, w, 3), 252, dtype=np.uint8)
    for c in range(3):
        img[:, :, c] = (img[:, :, c] * (1.0 - cloud_mask) + cloud_layer[:, :, c] * cloud_mask).astype(np.uint8)

    img = cv2.addWeighted(img, 0.82, np.full_like(img, 235), 0.18, 0)
    return img

def generate_all_samples(dest_dir):
    """Generate all 4 sample scenes and save to disk."""
    os.makedirs(dest_dir, exist_ok=True)
    
    samples = {
        'agriculture.jpg': create_agriculture_scene(),
        'coastal.jpg': create_coastal_scene(),
        'urban.jpg': create_urban_scene(),
        'mountain.jpg': create_mountain_scene()
    }
    
    for filename, img in samples.items():
        path = os.path.join(dest_dir, filename)
        cv2.imwrite(path, img)
        print(f"Generated sample scene: {path}")

if __name__ == '__main__':
    target_dir = os.path.join(os.path.dirname(__file__), 'static', 'samples')
    generate_all_samples(target_dir)
