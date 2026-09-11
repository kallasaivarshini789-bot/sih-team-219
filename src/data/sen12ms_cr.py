"""
SEN12MS-CR PyTorch Dataset Handler
Loads paired cloudy optical, cloud-free optical target, and Sentinel-1 SAR imagery patches.
"""

import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, Any, Optional, List, Tuple

from src.data.preprocessing import normalize_optical, normalize_sar
from src.data.cloud_mask import CloudMaskDetector


class SEN12MSCRDataset(Dataset):
    """PyTorch Dataset for SEN12MS-CR / SEN12MS-CR-TS paired multi-modal satellite patches."""

    def __init__(
        self,
        file_list: List[str],
        transform: Optional[Any] = None,
        optical_max_val: float = 10000.0,
        cloud_threshold: float = 0.4
    ):
        """
        Args:
            file_list: List of paths to dataset patch files or metadata dictionaries.
            transform: Optional Albumentations spatial augmentations pipeline.
            optical_max_val: Reflectance scale factor for optical bands.
            cloud_threshold: Threshold for s2cloudless / spectral cloud mask detector.
        """
        self.file_list = file_list
        self.transform = transform
        self.optical_max_val = optical_max_val
        self.cloud_detector = CloudMaskDetector(threshold=cloud_threshold)

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item_path = self.file_list[idx]

        if isinstance(item_path, dict):
            # Synthetic / direct dict memory object
            data_dict = item_path
            cloudy_opt = data_dict["cloudy_optical"] # (C_opt, H, W)
            target_opt = data_dict["target_optical"] # (C_opt, H, W)
            sar_data = data_dict["sar"]               # (C_sar, H, W)
            patch_name = data_dict.get("patch_name", f"patch_{idx:05d}")
        else:
            # File path (.npz or .npy patch file)
            path_obj = Path(item_path)
            if path_obj.suffix == ".npz":
                loaded = np.load(path_obj)
                cloudy_opt = loaded["cloudy_optical"]
                target_opt = loaded["target_optical"]
                sar_data = loaded["sar"]
            else:
                raise ValueError(f"Unsupported file format: {path_obj.suffix}")

            patch_name = path_obj.stem

        # Preprocessing & Normalization
        cloudy_opt_norm = normalize_optical(cloudy_opt, max_val=self.optical_max_val)
        target_opt_norm = normalize_optical(target_opt, max_val=self.optical_max_val)
        sar_norm = normalize_sar(sar_data)

        # Detect Cloud Mask on Cloudy Optical
        cloud_mask = self.cloud_detector.detect_cloud_mask(cloudy_opt_norm) # (1, H, W)

        # Stack Input: Cloudy Optical (C_opt) + SAR (C_sar) -> (C_opt + C_sar, H, W)
        stacked_input = np.concatenate([cloudy_opt_norm, sar_norm], axis=0)

        # Optional Data Augmentations
        if self.transform is not None:
            # Re-order to (H, W, C) for Albumentations
            input_hwc = np.transpose(stacked_input, (1, 2, 0))
            target_hwc = np.transpose(target_opt_norm, (1, 2, 0))
            mask_hwc = np.transpose(cloud_mask, (1, 2, 0))

            augmented = self.transform(
                image=input_hwc,
                target=target_hwc,
                mask=mask_hwc
            )

            stacked_input = np.transpose(augmented["image"], (2, 0, 1))
            target_opt_norm = np.transpose(augmented["target"], (2, 0, 1))
            cloud_mask = np.transpose(augmented["mask"], (2, 0, 1))

        # Convert to PyTorch Tensors
        input_tensor = torch.from_numpy(stacked_input).float()
        target_tensor = torch.from_numpy(target_opt_norm).float()
        mask_tensor = torch.from_numpy(cloud_mask).float()

        return {
            "cloudy_input": input_tensor,   # (C_in, H, W) e.g., (6, 256, 256)
            "target_optical": target_tensor, # (C_out, H, W) e.g., (4, 256, 256)
            "cloud_mask": mask_tensor,      # (1, H, W)
            "patch_name": patch_name
        }
