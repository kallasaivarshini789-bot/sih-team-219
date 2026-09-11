"""
Geographic Region Splitting Module
Partitions SEN12MS-CR patch files by geographic ROI / continent IDs to avoid spatial data leakage.
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple
from src.config import load_config


def extract_roi_from_filename(filename: str) -> str:
    """Extract ROI identifier from SEN12MS-CR patch filename.
    
    Standard format: ROIsxxxx_xxxx_patch_xx.tif -> ROIsxxxx or ROI_xx
    Example: ROI_01_patch_001.tif -> ROI_01
    """
    basename = Path(filename).name
    parts = basename.split("_")
    if len(parts) >= 2:
        if parts[0].isupper() and "ROI" in parts[0]:
            return f"{parts[0]}_{parts[1]}"
        elif "ROI" in parts[0]:
            return parts[0]
    return "ROI_DEFAULT"


def get_geographic_splits(
    file_list: List[str],
    config_path: str = "configs/config.yaml"
) -> Dict[str, List[str]]:
    """Partition a list of dataset file paths into train, val, and test splits by geographic ROI.

    Args:
        file_list: List of file paths to partition.
        config_path: Path to YAML configuration specifying geographic splits.

    Returns:
        Dictionary containing 'train', 'val', and 'test' file path lists.
    """
    cfg = load_config(config_path)
    train_rois = set(cfg.geographic_splits.train_rois)
    val_rois = set(cfg.geographic_splits.val_rois)
    test_rois = set(cfg.geographic_splits.test_rois)

    splits = {
        "train": [],
        "val": [],
        "test": []
    }

    for filepath in file_list:
        roi = extract_roi_from_filename(filepath)
        
        if roi in train_rois:
            splits["train"].append(filepath)
        elif roi in val_rois:
            splits["val"].append(filepath)
        elif roi in test_rois:
            splits["test"].append(filepath)
        else:
            # Fallback based on deterministic hash of ROI name if ROI not explicitly listed
            roi_hash = hash(roi) % 10
            if roi_hash < 7:
                splits["train"].append(filepath)
            elif roi_hash < 9:
                splits["val"].append(filepath)
            else:
                splits["test"].append(filepath)

    return splits
