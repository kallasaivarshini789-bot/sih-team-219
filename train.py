import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import SatelliteCloudDataset, generate_dummy_dataset
from models import get_model
from utils.losses import CompositeLoss
from utils.metrics import compute_metrics

def parse_args():
    parser = argparse.ArgumentParser(description="Train Real-time Satellite Cloud Removal Models")
    parser.add_argument("--model", type=str, default="attention_unet", choices=["unet", "attention_unet", "pix2pix"], help="Model architecture")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--data_dir", type=str, default="dummy_data", help="Directory containing train/val data")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Directory to save model checkpoints")
    parser.add_argument("--use_composite_loss", action="store_true", default=True, help="Use composite L1+SSIM loss")
    return parser.parse_args()

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for cloudy_imgs, cloud_free_imgs in tqdm(loader, desc="Training", leave=False):
        cloudy_imgs, cloud_free_imgs = cloudy_imgs.to(device), cloud_free_imgs.to(device)
        
        optimizer.zero_grad()
        outputs = model(cloudy_imgs)
        loss = criterion(outputs, cloud_free_imgs)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * cloudy_imgs.size(0)
        
    return total_loss / len(loader.dataset)

def validate_epoch(model, loader, criterion, device):
    model.eval()
    val_loss = 0.0
    psnr_scores = []
    ssim_scores = []
    
    with torch.no_grad():
        for cloudy_imgs, cloud_free_imgs in tqdm(loader, desc="Validation", leave=False):
            cloudy_imgs, cloud_free_imgs = cloudy_imgs.to(device), cloud_free_imgs.to(device)
            outputs = model(cloudy_imgs)
            loss = criterion(outputs, cloud_free_imgs)
            val_loss += loss.item() * cloudy_imgs.size(0)
            
            # Calculate validation PSNR & SSIM per batch
            for p, t in zip(outputs, cloud_free_imgs):
                m = compute_metrics(p, t)
                psnr_scores.append(m['PSNR'])
                ssim_scores.append(m['SSIM'])
                
    avg_loss = val_loss / len(loader.dataset)
    avg_psnr = float(sum(psnr_scores) / len(psnr_scores)) if psnr_scores else 0.0
    avg_ssim = float(sum(ssim_scores) / len(ssim_scores)) if ssim_scores else 0.0
    return avg_loss, avg_psnr, avg_ssim

def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    
    if not os.path.exists(args.data_dir):
        print(f"Data directory '{args.data_dir}' not found. Generating synthetic procedural satellite dataset...")
        generate_dummy_dataset(args.data_dir, num_samples=24)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Initializing {args.model.upper()} training pipeline on device: {device}")

    # Datasets and Loaders
    train_dataset = SatelliteCloudDataset(args.data_dir, split='train')
    val_dataset = SatelliteCloudDataset(args.data_dir, split='val')
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Model
    model = get_model(args.model).to(device)
    
    # Loss & Optimizer
    if args.use_composite_loss:
        criterion = CompositeLoss(lambda_l1=1.0, lambda_ssim=0.5, lambda_perceptual=0.1)
    else:
        criterion = nn.L1Loss()
        
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = float('inf')
    save_path = os.path.join(args.save_dir, f"{args.model}_best.pth")
    default_save_path = os.path.join(args.save_dir, "best_model.pth")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_psnr, val_ssim = validate_epoch(model, val_loader, criterion, device)
        
        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val PSNR: {val_psnr:.2f} dB | Val SSIM: {val_ssim:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            torch.save(model.state_dict(), default_save_path)
            print(f" -> Saved new best model checkpoint to '{save_path}'")

    print(f"\nTraining pipeline completed successfully! Model checkpoint saved.")

if __name__ == '__main__':
    main()
