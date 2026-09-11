"""
Cloud Detection Module for Sentinel-2 Optical Imagery
Provides s2cloudless, SCL band thresholding, and spectral cloud mask extraction.
"""

import numpy as np
from typing import Optional, Union


class CloudMaskDetector:
    """Cloud and cloud-shadow mask extraction for Sentinel-2 optical imagery.
    
    Supports:
    - s2cloudless gradient-boosted decision tree algorithm
    - Sentinel-2 Scene Classification Layer (SCL band)
    - Normalized Difference Cloud Index (NDCI / Reflectance threshold fallback)
    """

    def __init__(
        self,
        method: str = "s2cloudless",
        threshold: float = 0.4,
        dilation_pixels: int = 2
    ):
        """
        Args:
            method: Detection algorithm ('s2cloudless', 'scl', or 'threshold').
            threshold: Probability threshold for cloud classification (0.0 to 1.0).
            dilation_pixels: Morphological expansion pixels around detected clouds.
        """
        self.method = method.lower()
        self.threshold = threshold
        self.dilation_pixels = dilation_pixels
        self._s2cloudless_detector = None

        if self.method == "s2cloudless":
            try:
                from s2cloudless import S2PixelCloudDetector
                self._s2cloudless_detector = S2PixelCloudDetector(
                    threshold=self.threshold,
                    average_over=1,
                    dilation_size=self.dilation_pixels
                )
            except ImportError:
                # Graceful fallback to threshold method if s2cloudless package is not installed
                print("[CloudMaskDetector] s2cloudless library not installed. Falling back to spectral threshold method.")
                self.method = "threshold"

    def detect_cloud_mask(
        self,
        optical_data: np.ndarray,
        scl_band: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Compute binary cloud mask (1.0 = Cloud/Shadow, 0.0 = Clear).

        Args:
            optical_data: Optical array of shape (C, H, W) or (H, W, C) normalized [0.0, 1.0].
            scl_band: Optional Sentinel-2 SCL band of shape (H, W).

        Returns:
            Binary float32 cloud mask of shape (1, H, W) where 1.0 indicates cloud/shadow.
        """
        # Ensure shape is (H, W, C) for s2cloudless or (C, H, W) internally
        if optical_data.ndim == 3:
            if optical_data.shape[0] in [4, 13]:  # (C, H, W)
                c, h, w = optical_data.shape
                data_hwc = np.transpose(optical_data, (1, 2, 0))
            else:
                h, w, c = optical_data.shape
                data_hwc = optical_data
        else:
            raise ValueError(f"Expected 3D optical array, got shape {optical_data.shape}")

        if self.method == "scl" and scl_band is not None:
            # SCL band cloud classes: 3 = Cloud Shadow, 8 = Cloud Medium Prob, 9 = Cloud High Prob, 10 = Cirrus
            cloud_mask = np.isin(scl_band, [3, 8, 9, 10]).astype(np.float32)
            return np.expand_index(cloud_mask, axis=0) if cloud_mask.ndim == 2 else cloud_mask

        if self.method == "s2cloudless" and self._s2cloudless_detector is not None:
            # s2cloudless expects 10 or 13 band Sentinel-2 L1C input scaled 0-10000
            # If 4-band input, scale to s2cloudless shape or run prediction
            prob_map = self._s2cloudless_detector.get_cloud_probability_maps(
                np.expand_dims(data_hwc * 10000.0, axis=0)
            )[0]
            binary_mask = (prob_map > self.threshold).astype(np.float32)
            return np.expand_dims(binary_mask, axis=0)

        # Spectral / Brightness thresholding fallback
        # High reflectance in Blue (B02) + Red (B04) + NIR (B08) indicates clouds
        if data_hwc.shape[-1] >= 4:
            blue = data_hwc[:, :, 0]
            red = data_hwc[:, :, 2]
            nir = data_hwc[:, :, 3]
            
            # Brightness score and NIR/Red ratio
            brightness = (blue + red + nir) / 3.0
            is_bright = brightness > (self.threshold * 0.75)
            binary_mask = is_bright.astype(np.float32)
        else:
            brightness = np.mean(data_hwc, axis=-1)
            binary_mask = (brightness > self.threshold).astype(np.float32)

        return np.expand_dims(binary_mask, axis=0)
