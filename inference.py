import os
import torch
import numpy as np
from PIL import Image
import cv2

from models import get_model
from utils.cloud_mask import detect_clouds
from utils.postprocessing import blend_restored_image

class CloudRemover:
    def __init__(self, model_type="attention_unet", model_path="checkpoints/best_model.pth", device=None):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model_type = model_type
        
        self.model = get_model(model_type)
        
        # Check specific model checkpoint first, then fallback to general default
        specific_ckpt = os.path.join("checkpoints", f"{model_type}_best.pth")
        target_path = specific_ckpt if os.path.exists(specific_ckpt) else model_path

        if os.path.exists(target_path):
            try:
                self.model.load_state_dict(torch.load(target_path, map_location=self.device))
                print(f"Loaded {model_type.upper()} weights from '{target_path}'")
            except Exception as e:
                print(f"Warning loading '{target_path}': {e}. Using initialized model weights.")
        else:
            print(f"Notice: No checkpoint file found at '{target_path}'. Operating with initialized weights.")
            
        self.model.to(self.device)
        self.model.eval()

    def process_image(self, input_image: Image.Image, enable_postprocessing=True, feather_radius=7):
        """
        Process a cloudy optical satellite image.
        
        Returns:
            dict containing:
                - 'restored_image': PIL Image (restored cloud-free output)
                - 'raw_output': PIL Image (raw model prediction)
                - 'cloud_mask': PIL Image (binary cloud mask)
                - 'cloud_overlay': PIL Image (heatmap overlay)
                - 'cloud_pct': float (% cloud cover)
        """
        orig_img = input_image.convert('RGB')
        orig_np = np.array(orig_img)
        h, w = orig_np.shape[:2]

        # Resize for model processing (256x256)
        resized_np = cv2.resize(orig_np, (256, 256))
        
        # Prepare tensor
        tensor_in = torch.from_numpy(resized_np.transpose((2, 0, 1))).float() / 255.0
        tensor_in = tensor_in.unsqueeze(0).to(self.device)

        # Forward pass
        with torch.no_grad():
            tensor_out = self.model(tensor_in)

        # Convert back to numpy (256, 256, 3) uint8
        out_np = tensor_out.squeeze(0).cpu().numpy().transpose((1, 2, 0))
        out_np = (out_np * 255.0).clip(0, 255).astype(np.uint8)
        
        # Resize raw prediction back to original image dimensions
        raw_pred_np = cv2.resize(out_np, (w, h))

        # Cloud mask detection
        cloud_mask, cloud_pct, overlay_np = detect_clouds(orig_np)

        # Optional seamless histogram-matched blending
        if enable_postprocessing:
            final_np = blend_restored_image(orig_np, raw_pred_np, cloud_mask, feather_radius=feather_radius)
        else:
            final_np = raw_pred_np

        return {
            "restored_image": Image.fromarray(final_np),
            "raw_output": Image.fromarray(raw_pred_np),
            "cloud_mask": Image.fromarray(cloud_mask),
            "cloud_overlay": Image.fromarray(overlay_np),
            "cloud_pct": cloud_pct
        }

if __name__ == "__main__":
    remover = CloudRemover(model_type="attention_unet")
    test_img = Image.fromarray(np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8))
    res = remover.process_image(test_img)
    print("Inference test complete. Detected cloud coverage:", res["cloud_pct"], "%")
