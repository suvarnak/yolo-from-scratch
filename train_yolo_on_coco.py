import torch
import torch.nn as nn
import yaml
import os
from tqdm import tqdm
from data.data import Dataset
from torch.utils.data import DataLoader
from models.model import MyYolo
from utils import util
import argparse

# Example usage: update these paths to match your actual dataset
IMAGE_DIR = r'C:\Users\suvar\workspace\data\coco\train2017\images'
LABEL_DIR = r'C:\Users\suvar\workspace\data\coco\train2017\labels'
INPUT_SIZE = 640
AUGMENT = False
PARAMS = {
    'mosaic': 0.0,
    'mix_up': 0.0,
    'flip_ud': 0.0,
    'flip_lr': 0.0,
    'hsv_h': 0.015,
    'hsv_s': 0.7,
    'hsv_v': 0.4,
    'degrees': 0.0,
    'scale': 0.0,
    'shear': 0.0,
    'translate': 0.0
}

def main():
 
    parser = argparse.ArgumentParser(description='YOLO from Scratch Training')
    parser.add_argument('--version', type=str, default='n', help='YOLO model version: n/s/m/l/x')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--input-size', type=int, default=640, help='Input image size')
    parser.add_argument('--data-dir', type=str, default=r'C:\\Users\\suvar\\workspace\\data\\coco\\', help='COCO data directory')
    parser.add_argument('--train-list', type=str, default='instances_train2017.txt', help='COCO train image list')

    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume training')
    args = parser.parse_args()

    print(f"\nSample commands:")
    print("python train_yolo_on_coco.py --version n --epochs 100 --batch-size 16")
    print("python train_yolo_on_coco.py --version s --epochs 200 --batch-size 32")
    print("python train_yolo_on_coco.py --version l --epochs 300 --batch-size 8 --input-size 1024")
    print()

    # Check CUDA
    if not torch.cuda.is_available():
        print("CUDA not available. Training requires GPU.")
        exit(1)

    device = torch.device('cuda')
    print(f"Using device: {torch.cuda.get_device_name()}")

    # Configuration
    data_dir = args.data_dir
    input_size = args.input_size
    batch_size = args.batch_size
    num_epochs = args.epochs
    save_dir = 'runs/train'
    os.makedirs(save_dir, exist_ok=True)

    # Load params
    with open('utils/hyperparams.yaml', errors='ignore') as f:
        params = yaml.safe_load(f)

    # Use dataloader logic from test_dataloader.py
    # IMAGE_DIR = os.path.join(data_dir, 'train2017', 'images')
    # INPUT_SIZE = input_size
    PARAMS = params
    AUGMENT = True
    if not os.path.exists(IMAGE_DIR):
        print(f"ERROR: Image directory '{IMAGE_DIR}' does not exist.")
        return
    image_files = [os.path.join(IMAGE_DIR, fname) for fname in os.listdir(IMAGE_DIR)
                  if os.path.splitext(fname)[-1].lower() in ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff')]
    print(f"Found {len(image_files)} images.")
    if not image_files:
        print("ERROR: No image files found in the directory.")
        return
    train_data = Dataset(image_files, INPUT_SIZE, PARAMS, AUGMENT)
    print(f"Loaded {len(train_data)} samples.")
    print("First 5 label shapes:", [lbl.shape for lbl in train_data.labels[:5]])
    print("First 5 labels:", [lbl for lbl in train_data.labels[:5]])    
    train_loader = DataLoader(
        train_data, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=4, 
        pin_memory=True, 
        collate_fn=Dataset.collate_fn
    )


    # Model
    model = MyYolo(version=args.version).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    # Loss and optimizer
    criterion = util.ComputeLoss(model, params)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Resume from checkpoint if specified
    start_epoch = 0
    best_loss = float('inf')
    if args.resume is not None and os.path.isfile(args.resume):
        print(f"Resuming training from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_loss = checkpoint.get('loss', float('inf'))
        print(f"Resumed at epoch {start_epoch}, best loss {best_loss:.4f}")

    # Training loop
    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for batch_idx, (imgs, targets) in enumerate(pbar):
            imgs = imgs.float().to(device)
            targets = {k: v.to(device) for k, v in targets.items()}
            # Forward pass
            outputs = model(imgs)
            loss_components = criterion(outputs, targets)
            if batch_idx == 0:
                print("\n[Diagnostics] First batch of epoch:")
                # ...diagnostics removed...
                for i, out in enumerate(outputs):
                    print(f"  Output[{i}] stats: mean={out.mean().item():.4f}, std={out.std().item():.4f}, shape={tuple(out.shape)}")
                print("  Sample target:", {k: v[0].cpu().numpy() if hasattr(v, 'cpu') and v.shape[0] > 0 else None for k, v in targets.items()})
                print("  Loss components:")
                for i, lc in enumerate(loss_components):
                    print(f"    Component {i}: value={lc.item():.4f}, type={type(lc)}")
            loss = torch.cat([lc.view(-1) for lc in loss_components]).sum()
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
