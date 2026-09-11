import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np
import cv2

class SatelliteCloudDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        """
        Args:
            root_dir (string): Directory with images containing 'cloudy' and 'cloud_free' subdirectories.
            split (string): 'train', 'val', or 'test'.
            transform (callable, optional): Albumentations or custom transform.
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        
        self.cloudy_dir = os.path.join(root_dir, split, 'cloudy')
        self.cloud_free_dir = os.path.join(root_dir, split, 'cloud_free')
        
        if os.path.exists(self.cloudy_dir) and os.path.exists(self.cloud_free_dir):
            self.image_names = sorted(os.listdir(self.cloudy_dir))
        else:
            self.image_names = []

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.image_names[idx]
        cloudy_img_path = os.path.join(self.cloudy_dir, img_name)
        cloud_free_img_path = os.path.join(self.cloud_free_dir, img_name)

        cloudy_image = Image.open(cloudy_img_path).convert('RGB')
        cloud_free_image = Image.open(cloud_free_img_path).convert('RGB')
        
        cloudy_np = np.array(cloudy_image)
        cloud_free_np = np.array(cloud_free_image)

        if self.transform:
            transformed = self.transform(image=cloudy_np, target=cloud_free_np)
            cloudy_np = transformed['image']
            cloud_free_np = transformed['target']
            
            # Convert HWC to CHW
            cloudy_tensor = torch.from_numpy(cloudy_np.transpose((2, 0, 1))).float() / 255.0
            cloud_free_tensor = torch.from_numpy(cloud_free_np.transpose((2, 0, 1))).float() / 255.0
            return cloudy_tensor, cloud_free_tensor
        else:
            cloudy_tensor = torch.from_numpy(cloudy_np.transpose((2, 0, 1))).float() / 255.0
            cloud_free_tensor = torch.from_numpy(cloud_free_np.transpose((2, 0, 1))).float() / 255.0
            return cloudy_tensor, cloud_free_tensor


def generate_procedural_cloud(img_size=(256, 256)):
    """Generate realistic procedural cloud alpha mask using multi-scale noise."""
    h, w = img_size
    # Create smooth multi-octave noise using Gaussian blurring
    base_noise = np.random.uniform(0, 1, (h // 4, w // 4)).astype(np.float32)
    base_noise = cv2.resize(base_noise, (w, h), interpolation=cv2.INTER_CUBIC)
    
    fine_noise = np.random.uniform(0, 1, (h // 2, w // 2)).astype(np.float32)
    fine_noise = cv2.resize(fine_noise, (w, h), interpolation=cv2.INTER_CUBIC)

    combined = 0.7 * base_noise + 0.3 * fine_noise
    combined = cv2.GaussianBlur(combined, (21, 21), 0)
    
    # Threshold & normalize to create dense cloud core with soft wispy edges
    cloud_mask = np.clip((combined - 0.35) * 3.0, 0.0, 1.0)
    return cloud_mask


def generate_dummy_dataset(base_dir, num_samples=20, img_size=(256, 256)):
    """Generate synthetic satellite dataset with realistic procedural cloud cover."""
    for split in ['train', 'val']:
        os.makedirs(os.path.join(base_dir, split, 'cloudy'), exist_ok=True)
        os.makedirs(os.path.join(base_dir, split, 'cloud_free'), exist_ok=True)
        
        n_imgs = num_samples if split == 'train' else max(5, num_samples // 4)
        for i in range(n_imgs):
            # Generate realistic terrain background (green/earth tones)
            terrain = np.zeros((img_size[0], img_size[1], 3), dtype=np.uint8)
            terrain[:, :, 0] = np.random.randint(30, 90, img_size, dtype=np.uint8)   # Red
            terrain[:, :, 1] = np.random.randint(100, 180, img_size, dtype=np.uint8) # Green (Vegetation)
            terrain[:, :, 2] = np.random.randint(40, 110, img_size, dtype=np.uint8)  # Blue

            # Add structured ground patterns (river or road)
            y, x = np.ogrid[:img_size[0], :img_size[1]]
            river_mask = np.abs(y - (0.5 * x + 50)) < 8
            terrain[river_mask] = [30, 80, 200]  # Blue water

            # Generate procedural cloud overlay
            cloud_alpha = generate_procedural_cloud(img_size)
            cloud_alpha_3d = np.expand_dims(cloud_alpha, axis=-1)

            # Cloud color (bright white/greyish)
            cloud_color = np.random.randint(220, 255, (img_size[0], img_size[1], 3), dtype=np.uint8)

            cloudy_img = ((1.0 - cloud_alpha_3d) * terrain + cloud_alpha_3d * cloud_color).astype(np.uint8)

            img_name = f"sat_scene_{i:04d}.png"
            Image.fromarray(terrain).save(os.path.join(base_dir, split, 'cloud_free', img_name))
            Image.fromarray(cloudy_img).save(os.path.join(base_dir, split, 'cloudy', img_name))

    print(f"Procedural synthetic dataset successfully generated at '{base_dir}'.")

if __name__ == '__main__':
    generate_dummy_dataset('dummy_data', num_samples=5)
