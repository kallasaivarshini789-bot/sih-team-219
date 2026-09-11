"""
SEN12MS-CR Sample Synthesizer & Downloader Helper Script
Creates realistic multi-modal paired patch files (.npz) for testing data loading and preprocessing.
"""

import os
from pathlib import Path
import numpy as np
from tqdm import tqdm


def generate_synthetic_patches(
    output_dir: str = "dummy_data",
    num_rois: int = 12,
    patches_per_roi: int = 4,
    patch_size: int = 256
) -> None:
    """Generate realistic synthetic SEN12MS-CR paired optical + SAR patches.

    Args:
        output_dir: Directory where generated .npz patch files will be saved.
        num_rois: Number of geographic ROIs to simulate.
        patches_per_roi: Number of patches per ROI.
        patch_size: Height/width of each patch.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"[Sample Generator] Synthesizing SEN12MS-CR sample patches in '{output_dir}'...")

    total_created = 0
    for roi_idx in range(1, num_rois + 1):
        roi_name = f"ROI_{roi_idx:02d}"
        
        for patch_idx in range(1, patches_per_roi + 1):
            patch_name = f"{roi_name}_patch_{patch_idx:03d}.npz"
            filepath = out_path / patch_name

            # 1. Generate Target Optical (Cloud-Free ground truth) - 4 channels (B02 Blue, B03 Green, B04 Red, B08 NIR)
            # Base land cover terrain pattern with reflectance values [1000, 4000]
            xx, yy = np.meshgrid(np.linspace(0, 4 * np.pi, patch_size), np.linspace(0, 4 * np.pi, patch_size))
            terrain = (np.sin(xx) * np.cos(yy) + 1.0) * 1500.0 + 1000.0 # [1000, 4000]
            
            target_optical = np.zeros((4, patch_size, patch_size), dtype=np.float32)
            target_optical[0] = terrain * 0.8  # Blue
            target_optical[1] = terrain * 1.0  # Green
            target_optical[2] = terrain * 0.9  # Red
            target_optical[3] = terrain * 1.5  # NIR (Vegetation high response)

            # 2. Generate Cloud Layer (Synthetic Gaussian cloud puff)
            cloud_center_x = np.random.randint(50, patch_size - 50)
            cloud_center_y = np.random.randint(50, patch_size - 50)
            cloud_grid_y, cloud_grid_x = np.ogrid[:patch_size, :patch_size]
            dist_from_center = np.sqrt((cloud_grid_x - cloud_center_x)**2 + (cloud_grid_y - cloud_center_y)**2)
            
            cloud_intensity = np.exp(-dist_from_center**2 / (2 * (40.0**2))) * 6000.0 # High reflectance cloud bump
            
            # Add noise to clouds
            cloud_noise = np.random.normal(0.0, 500.0, size=(patch_size, patch_size))
            cloud_layer = np.maximum(0.0, cloud_intensity + cloud_noise)

            cloudy_optical = np.clip(target_optical + cloud_layer, 0.0, 10000.0).astype(np.float32)

            # 3. Generate Sentinel-1 SAR Channels (VV & VH linear power)
            # SAR penetrates cloud layer, so it correlates with terrain structural roughness
            sar_vv = (terrain * 0.005 + np.random.exponential(0.02, size=(patch_size, patch_size))).astype(np.float32)
            sar_vh = (terrain * 0.001 + np.random.exponential(0.005, size=(patch_size, patch_size))).astype(np.float32)
            sar_data = np.stack([sar_vv, sar_vh], axis=0) # (2, H, W)

            # Save as .npz dictionary
            np.savez_compressed(
                filepath,
                cloudy_optical=cloudy_optical,
                target_optical=target_optical,
                sar=sar_data
            )
            total_created += 1

    print(f"[Sample Generator] Successfully generated {total_created} synthetic patch files across {num_rois} ROIs.")


if __name__ == "__main__":
    generate_synthetic_patches()
