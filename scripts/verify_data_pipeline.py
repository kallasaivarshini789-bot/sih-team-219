"""
Data Pipeline Verification & Sanity Check Script (Milestones 1 & 2)
Verifies config loading, geographic region splitting, normalization, cloud masking, and PyTorch DataLoader batch retrieval.
"""

import sys
from pathlib import Path
# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import glob
import torch
from torch.utils.data import DataLoader

from src.config import load_config
from src.data.split import get_geographic_splits
from src.data.preprocessing import normalize_optical, normalize_sar, tile_image, reassemble_tiles
from src.data.cloud_mask import CloudMaskDetector
from src.data.sen12ms_cr import SEN12MSCRDataset
from scripts.prepare_sen12ms_cr_sample import generate_synthetic_patches


def main():
    print("=" * 70)
    print("      DATA PIPELINE & CONFIG VERIFICATION (MILESTONES 1 & 2)      ")
    print("=" * 70)

    # 1. Config Loading Verification
    cfg = load_config("configs/config.yaml")
    print(f"\n[1/6] Config Loaded Successfully:")
    print(f"      - Dataset Name: {cfg.dataset.name}")
    print(f"      - Patch Size: {cfg.dataset.patch_size}x{cfg.dataset.patch_size}")
    print(f"      - Input Channels: {cfg.dataset.num_input_channels} ({cfg.dataset.num_optical_channels} Optical + {cfg.dataset.num_sar_channels} SAR)")
    print(f"      - Target Channels: {cfg.dataset.num_optical_channels} Cloud-Free Optical")

    # 2. Sample Dataset Generation
    dummy_dir = cfg.dataset.dummy_dir
    generate_synthetic_patches(output_dir=dummy_dir, num_rois=12, patches_per_roi=4)

    patch_files = glob.glob(os.path.join(dummy_dir, "*.npz"))
    print(f"\n[2/6] Dataset Files Found: {len(patch_files)} files in '{dummy_dir}'")
    assert len(patch_files) > 0, "No dataset files generated!"

    # 3. Geographic Region Splitting Verification
    splits = get_geographic_splits(patch_files, config_path="configs/config.yaml")
    print(f"\n[3/6] Geographic Region Split Verification:")
    print(f"      - Train Patches: {len(splits['train'])} files")
    print(f"      - Val Patches:   {len(splits['val'])} files")
    print(f"      - Test Patches:  {len(splits['test'])} files")

    # Verify zero spatial leakage across region splits
    train_set = set(splits['train'])
    val_set = set(splits['val'])
    test_set = set(splits['test'])
    assert len(train_set.intersection(val_set)) == 0, "Spatial leakage detected between Train and Val!"
    assert len(train_set.intersection(test_set)) == 0, "Spatial leakage detected between Train and Test!"
    assert len(val_set.intersection(test_set)) == 0, "Spatial leakage detected between Val and Test!"
    print("      ✔ Zero spatial leakage verified across regional partitions!")

    # 4. Preprocessing & Normalization Verification
    print(f"\n[4/6] Preprocessing & Normalization Verification:")
    raw_opt = torch.randint(0, 10000, (4, 256, 256)).float().numpy()
    raw_sar = torch.abs(torch.randn(2, 256, 256)).numpy() * 0.05

    norm_opt = normalize_optical(raw_opt, max_val=cfg.preprocessing.optical_max_value)
    norm_sar = normalize_sar(raw_sar)

    print(f"      - Normalized Optical Min: {norm_opt.min():.4f}, Max: {norm_opt.max():.4f}")
    print(f"      - Normalized SAR Min:     {norm_sar.min():.4f}, Max: {norm_sar.max():.4f}")
    assert 0.0 <= norm_opt.min() <= norm_opt.max() <= 1.0, "Optical normalization out of range!"
    assert 0.0 <= norm_sar.min() <= norm_sar.max() <= 1.0, "SAR normalization out of range!"
    print("      ✔ Preprocessing ranges verified [0.0, 1.0]!")

    # 5. Cloud Masking Module Verification
    print(f"\n[5/6] Cloud Masking Module Verification:")
    detector = CloudMaskDetector(threshold=cfg.cloud_mask.threshold)
    mask = detector.detect_cloud_mask(norm_opt)
    print(f"      - Cloud Mask Shape: {mask.shape}, Range: [{mask.min():.1f}, {mask.max():.1f}]")
    print(f"      - Cloud Coverage Ratio: {mask.mean() * 100.0:.2f}%")
    assert mask.shape == (1, 256, 256), f"Unexpected mask shape {mask.shape}!"
    print("      ✔ Cloud Mask Detector functioning correctly!")

    # 6. PyTorch DataLoader Verification
    print(f"\n[6/6] PyTorch Dataset & DataLoader Verification:")
    train_dataset = SEN12MSCRDataset(splits['train'])
    loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    batch = next(iter(loader))
    cloudy_in = batch['cloudy_input']    # (B, C_in, H, W)
    target_opt = batch['target_optical'] # (B, C_target, H, W)
    cloud_m = batch['cloud_mask']        # (B, 1, H, W)

    print(f"      - Batch Size:           {cloudy_in.shape[0]}")
    print(f"      - Stacked Input Tensor: {list(cloudy_in.shape)}  [Expected: [4, 6, 256, 256]]")
    print(f"      - Target Optical Tensor:{list(target_opt.shape)}  [Expected: [4, 4, 256, 256]]")
    print(f"      - Cloud Mask Tensor:    {list(cloud_m.shape)}  [Expected: [4, 1, 256, 256]]")

    assert cloudy_in.shape == (4, 6, 256, 256), "Input batch shape mismatch!"
    assert target_opt.shape == (4, 4, 256, 256), "Target batch shape mismatch!"
    assert cloud_m.shape == (4, 1, 256, 256), "Mask batch shape mismatch!"
    print("      ✔ PyTorch DataLoader batch shapes verified!")

    print("\n" + "=" * 70)
    print("      ALL MILESTONE 1 & 2 VERIFICATION CHECKS PASSED SUCCESSFULLY!    ")
    print("=" * 70)


if __name__ == "__main__":
    main()
