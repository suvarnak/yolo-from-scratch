import torch
import torch.nn as nn
import yaml
import os
from tqdm import tqdm
from data.data import Dataset
from torch.utils.data import DataLoader
from models.model import MyYolo
from utils import util

def main():
    # Check CUDA
    if not torch.cuda.is_available():
        print("CUDA not available. Training requires GPU.")
        exit(1)

    device = torch.device('cuda')
    print(f"Using device: {torch.cuda.get_device_name()}")

    # Configuration
    data_dir = 'C:/Users/suvar/workspace/data/coco'
    input_size = 640
    batch_size = 16  # Reduced for stability
    num_epochs = 5
    save_dir = 'runs/train'
    os.makedirs(save_dir, exist_ok=True)

    # Load filenames
    filenames_train = []
    with open(f'{data_dir}/sanity_train.txt') as f:
        for line in f:
            filename = line.strip()
            img_path = f'{data_dir}/train2017/images/{filename}'
            if os.path.isfile(img_path):
                filenames_train.append(img_path)
            else:
                print(f"Warning: Image file not found: {img_path}")

    # Load params
    with open('utils/hyperparams.yaml', errors='ignore') as f:
        params = yaml.safe_load(f)

    # Dataset and DataLoader
    train_data = Dataset(filenames_train, input_size, params, augment=True)
    train_loader = DataLoader(
        train_data, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=4, 
        pin_memory=True, 
        collate_fn=Dataset.collate_fn
    )

    # Model
    model = MyYolo(version='n').to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    # Loss and optimizer
    criterion = util.ComputeLoss(model, params)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Training loop
    best_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for batch_idx, (imgs, targets) in enumerate(pbar):
            imgs = imgs.float().to(device)
            targets = {k: v.to(device) for k, v in targets.items()}
            
            # Forward pass
            outputs = model(imgs)
            loss = sum(criterion(outputs, targets))
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            
            running_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Update learning rate
        scheduler.step()
        
        # Calculate average loss
        avg_loss = running_loss / len(train_loader)
        print(f'Epoch {epoch+1}: Loss={avg_loss:.4f}, LR={optimizer.param_groups[0]["lr"]:.6f}')
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, f'{save_dir}/best.pt')
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, f'{save_dir}/epoch_{epoch+1}.pt')

    print("Training completed!")

if __name__ == "__main__":
    main()
