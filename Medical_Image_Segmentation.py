"""
Medical Image Segmentation using U-Net
For organ/tumor segmentation in CT/MRI scans
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import cv2
from tqdm import tqdm

# Install: pip install torch torchvision opencv-python scikit-learn tqdm


class DoubleConv(nn.Module):
    """Double convolution block for U-Net"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)


class UNet(nn.Module):
    """U-Net architecture for medical image segmentation"""
    
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Encoder
        for feature in features:
            self.encoder.append(DoubleConv(in_channels, feature))
            in_channels = feature
        
        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        
        # Decoder
        for feature in reversed(features):
            self.decoder.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.decoder.append(DoubleConv(feature * 2, feature))
        
        # Final output
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
    
    def forward(self, x):
        skip_connections = []
        
        # Encoder
        for encode in self.encoder:
            x = encode(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        
        # Decoder
        for idx in range(0, len(self.decoder), 2):
            x = self.decoder[idx](x)
            skip_connection = skip_connections[idx // 2]
            
            # Handle size mismatch
            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:])
            
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.decoder[idx + 1](concat_skip)
        
        return torch.sigmoid(self.final_conv(x))


class MedicalImageDataset(Dataset):
    """Dataset for medical images and masks"""
    
    def __init__(self, images, masks, transform=None):
        self.images = images
        self.masks = masks
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]
        
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        
        # Ensure correct shape [C, H, W]
        if len(image.shape) == 2:
            image = image[np.newaxis, ...]
        if len(mask.shape) == 2:
            mask = mask[np.newaxis, ...]
        
        return torch.FloatTensor(image), torch.FloatTensor(mask)


class DiceLoss(nn.Module):
    """Dice Loss for segmentation"""
    
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = pred.view(-1)
        target = target.view(-1)
        
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        
        return 1 - dice


class CombinedLoss(nn.Module):
    """Combined Dice and BCE loss"""
    
    def __init__(self, dice_weight=0.5, bce_weight=0.5):
        super().__init__()
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCELoss()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
    
    def forward(self, pred, target):
        dice = self.dice_loss(pred, target)
        bce = self.bce_loss(pred, target)
        return self.dice_weight * dice + self.bce_weight * bce


def dice_coefficient(pred, target, threshold=0.5):
    """Calculate Dice coefficient metric"""
    pred = (pred > threshold).float()
    target = target.float()
    
    intersection = (pred * target).sum()
    dice = (2. * intersection) / (pred.sum() + target.sum() + 1e-8)
    
    return dice.item()


def iou_score(pred, target, threshold=0.5):
    """Calculate Intersection over Union"""
    pred = (pred > threshold).float()
    target = target.float()
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    iou = intersection / (union + 1e-8)
    
    return iou.item()


class MedicalSegmentationTrainer:
    """Training pipeline for medical image segmentation"""
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.train_losses = []
        self.val_losses = []
        self.train_dice = []
        self.val_dice = []
    
    def train_epoch(self, train_loader, optimizer, criterion):
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0
        epoch_dice = 0
        
        for images, masks in tqdm(train_loader, desc="Training"):
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = self.model(images)
            loss = criterion(outputs, masks)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Metrics
            epoch_loss += loss.item()
            epoch_dice += dice_coefficient(outputs, masks)
        
        return epoch_loss / len(train_loader), epoch_dice / len(train_loader)
    
    def validate(self, val_loader, criterion):
        """Validate the model"""
        self.model.eval()
        epoch_loss = 0
        epoch_dice = 0
        epoch_iou = 0
        
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc="Validation"):
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                outputs = self.model(images)
                loss = criterion(outputs, masks)
                
                epoch_loss += loss.item()
                epoch_dice += dice_coefficient(outputs, masks)
                epoch_iou += iou_score(outputs, masks)
        
        n = len(val_loader)
        return epoch_loss / n, epoch_dice / n, epoch_iou / n
    
    def train(self, train_loader, val_loader, epochs=50, lr=1e-4):
        """Complete training loop"""
        criterion = CombinedLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=5, factor=0.5
        )
        
        best_dice = 0
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            # Train
            train_loss, train_dice = self.train_epoch(train_loader, optimizer, criterion)
            
            # Validate
            val_loss, val_dice, val_iou = self.validate(val_loader, criterion)
            
            # Update learning rate
            scheduler.step(val_loss)
            
            # Save metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_dice.append(train_dice)
            self.val_dice.append(val_dice)
            
            print(f"Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}, Val IoU: {val_iou:.4f}")
            
            # Save best model
            if val_dice > best_dice:
                best_dice = val_dice
                torch.save(self.model.state_dict(), 'best_unet_model.pth')
                print(f"Saved best model with Dice: {best_dice:.4f}")
    
    def plot_training_history(self):
        """Plot training curves"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss
        axes[0].plot(self.train_losses, label='Train Loss')
        axes[0].plot(self.val_losses, label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Dice coefficient
        axes[1].plot(self.train_dice, label='Train Dice')
        axes[1].plot(self.val_dice, label='Val Dice')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Dice Coefficient')
        axes[1].set_title('Training and Validation Dice Score')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.show()
    
    def predict(self, image):
        """Predict segmentation mask for single image"""
        self.model.eval()
        with torch.no_grad():
            if len(image.shape) == 2:
                image = image[np.newaxis, np.newaxis, ...]
            elif len(image.shape) == 3:
                image = image[np.newaxis, ...]
            
            image_tensor = torch.FloatTensor(image).to(self.device)
            output = self.model(image_tensor)
            return output.cpu().numpy()[0, 0]
    
    def visualize_predictions(self, images, masks, num_samples=4):
        """Visualize predictions on sample images"""
        fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))
        
        for i in range(num_samples):
            # Predict
            pred = self.predict(images[i])
            
            # Plot
            axes[i, 0].imshow(images[i].squeeze(), cmap='gray')
            axes[i, 0].set_title('Input Image')
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(masks[i].squeeze(), cmap='gray')
            axes[i, 1].set_title('Ground Truth')
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(pred, cmap='gray')
            axes[i, 2].set_title('Prediction')
            axes[i, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig('predictions.png')
        plt.show()


# Example usage and data generation
def generate_synthetic_data(n_samples=1000, img_size=256):
    """Generate synthetic medical images for demonstration"""
    images = []
    masks = []
    
    for _ in range(n_samples):
        # Create synthetic image with circles/ellipses
        img = np.zeros((img_size, img_size), dtype=np.float32)
        mask = np.zeros((img_size, img_size), dtype=np.float32)
        
        # Random organ-like shapes
        n_shapes = np.random.randint(1, 4)
        for _ in range(n_shapes):
            center = (np.random.randint(50, img_size-50), np.random.randint(50, img_size-50))
            axes = (np.random.randint(20, 60), np.random.randint(20, 60))
            angle = np.random.randint(0, 180)
            
            cv2.ellipse(img, center, axes, angle, 0, 360, 1.0, -1)
            cv2.ellipse(mask, center, axes, angle, 0, 360, 1.0, -1)
        
        # Add noise to image
        noise = np.random.normal(0, 0.1, img.shape)
        img = np.clip(img + noise, 0, 1)
        
        images.append(img)
        masks.append(mask)
    
    return np.array(images), np.array(masks)


if __name__ == "__main__":
    # Generate synthetic data
    print("Generating synthetic medical images...")
    images, masks = generate_synthetic_data(n_samples=500, img_size=256)
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        images, masks, test_size=0.2, random_state=42
    )
    
    # Create datasets and loaders
    train_dataset = MedicalImageDataset(X_train, y_train)
    val_dataset = MedicalImageDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    # Initialize model
    model = UNet(in_channels=1, out_channels=1, features=[64, 128, 256, 512])
    
    # Train
    trainer = MedicalSegmentationTrainer(model)
    print("\nStarting training...")
    trainer.train(train_loader, val_loader, epochs=30, lr=1e-3)
    
    # Plot results
    trainer.plot_training_history()
    trainer.visualize_predictions(X_val, y_val, num_samples=4)
    
    print("\nTraining complete! Best model saved as 'best_unet_model.pth'")