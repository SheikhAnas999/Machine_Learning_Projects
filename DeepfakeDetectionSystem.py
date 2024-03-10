"""
Deepfake Detection System
Uses CNN with attention mechanisms for detecting manipulated videos
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import seaborn as sns

# Install: pip install torch torchvision opencv-python scikit-learn seaborn


class SpatialAttention(nn.Module):
    """Spatial attention mechanism"""
    
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_pool, max_pool], dim=1)
        attention = self.sigmoid(self.conv(concat))
        return x * attention


class ChannelAttention(nn.Module):
    """Channel attention mechanism"""
    
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(in_channels // reduction, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        b, c, _, _ = x.size()
        avg = self.avg_pool(x).view(b, c)
        max_p = self.max_pool(x).view(b, c)
        
        avg_out = self.fc(avg)
        max_out = self.fc(max_p)
        
        attention = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return x * attention.expand_as(x)


class DeepfakeDetector(nn.Module):
    """CNN with attention for deepfake detection"""
    
    def __init__(self, pretrained=True, num_classes=2):
        super().__init__()
        
        # Use pretrained EfficientNet as backbone
        self.backbone = models.efficientnet_b0(pretrained=pretrained)
        
        # Get feature dimension
        in_features = self.backbone.classifier[1].in_features
        
        # Remove original classifier
        self.backbone.classifier = nn.Identity()
        
        # Add attention modules
        self.channel_attention = ChannelAttention(in_features)
        
        # Custom classifier with dropout
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        # Extract features
        features = self.backbone(x)
        
        # Apply attention
        features = features.view(features.size(0), features.size(1), 1, 1)
        features = self.channel_attention(features)
        features = features.view(features.size(0), -1)
        
        # Classify
        output = self.classifier(features)
        return output


class VideoFrameDataset(Dataset):
    """Dataset for video frames with real/fake labels"""
    
    def __init__(self, video_paths, labels, num_frames=10, transform=None):
        self.video_paths = video_paths
        self.labels = labels
        self.num_frames = num_frames
        self.transform = transform
        
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
    
    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        
        # Extract frames from video
        frames = self.extract_frames(video_path, self.num_frames)
        
        # Transform frames
        transformed_frames = []
        for frame in frames:
            transformed_frames.append(self.transform(frame))
        
        # Stack frames
        video_tensor = torch.stack(transformed_frames)
        
        return video_tensor, label
    
    def extract_frames(self, video_path, num_frames):
        """Extract evenly spaced frames from video"""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            # Return blank frames if video can't be read
            return [np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(num_frames)]
        
        # Calculate frame indices
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            else:
                frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
        
        cap.release()
        return frames


class FrameLevelDataset(Dataset):
    """Dataset for individual frames (simpler approach)"""
    
    def __init__(self, frames, labels, transform=None):
        self.frames = frames
        self.labels = labels
        self.transform = transform
        
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
    
    def __len__(self):
        return len(self.frames)
    
    def __getitem__(self, idx):
        frame = self.frames[idx]
        label = self.labels[idx]
        
        if self.transform:
            frame = self.transform(frame)
        
        return frame, label


class DeepfakeTrainer:
    """Training pipeline for deepfake detection"""
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
    
    def train_epoch(self, train_loader, optimizer, criterion):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for frames, labels in tqdm(train_loader, desc="Training"):
            frames = frames.to(self.device)
            labels = labels.to(self.device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(frames)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self, val_loader, criterion):
        """Validate the model"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for frames, labels in tqdm(val_loader, desc="Validation"):
                frames = frames.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(frames)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item()
                
                probs = F.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)
                
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())
        
        epoch_loss = running_loss / len(val_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc, all_preds, all_labels, all_probs
    
    def train(self, train_loader, val_loader, epochs=30, lr=1e-4):
        """Complete training loop"""
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', patience=3, factor=0.5
        )
        
        best_acc = 0
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, criterion)
            
            # Validate
            val_loss, val_acc, val_preds, val_labels, val_probs = self.validate(
                val_loader, criterion
            )
            
            # Update learning rate
            scheduler.step(val_acc)
            
            # Save metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Save best model
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(self.model.state_dict(), 'best_deepfake_detector.pth')
                print(f"Saved best model with accuracy: {best_acc:.2f}%")
        
        return val_preds, val_labels, val_probs
    
    def plot_training_history(self):
        """Plot training curves"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss
        axes[0].plot(self.train_losses, label='Train Loss', marker='o')
        axes[0].plot(self.val_losses, label='Val Loss', marker='s')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Accuracy
        axes[1].plot(self.train_accs, label='Train Acc', marker='o')
        axes[1].plot(self.val_accs, label='Val Acc', marker='s')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig('deepfake_training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_confusion_matrix(self, y_true, y_pred, class_names=['Real', 'Fake']):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_roc_curve(self, y_true, y_probs):
        """Plot ROC curve"""
        fpr, tpr, thresholds = roc_curve(y_true, y_probs)
        auc_score = roc_auc_score(y_true, y_probs)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.3f})', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - Deepfake Detection')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return auc_score
    
    def evaluate(self, y_true, y_pred, y_probs):
        """Complete evaluation with metrics"""
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, 
                                   target_names=['Real', 'Fake'],
                                   digits=4))
        
        # Plot confusion matrix
        self.plot_confusion_matrix(y_true, y_pred)
        
        # Plot ROC curve
        auc_score = self.plot_roc_curve(y_true, y_probs)
        
        print(f"\nAUC-ROC Score: {auc_score:.4f}")
        
    def predict_frame(self, frame):
        """Predict if a single frame is real or fake"""
        self.model.eval()
        
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        frame_tensor = transform(frame).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(frame_tensor)
            prob = F.softmax(output, dim=1)
            pred = output.argmax(1).item()
            confidence = prob[0, pred].item()
        
        return pred, confidence
    
    def predict_video(self, video_path, num_frames=30):
        """Predict if a video is real or fake"""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            return None, 0
        
        # Sample frames
        indices = np.linspace(0, total_frames - 1, min(num_frames, total_frames), dtype=int)
        
        predictions = []
        confidences = []
        
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pred, conf = self.predict_frame(frame)
                predictions.append(pred)
                confidences.append(conf)
        
        cap.release()
        
        # Aggregate predictions
        final_pred = int(np.round(np.mean(predictions)))
        final_conf = np.mean(confidences)
        
        return final_pred, final_conf


# Generate synthetic data for demonstration
def generate_synthetic_data(n_samples=1000, img_size=224):
    """Generate synthetic face-like images"""
    frames = []
    labels = []
    
    for i in range(n_samples):
        # Create synthetic image
        frame = np.random.randint(0, 255, (img_size, img_size, 3), dtype=np.uint8)
        
        # Add face-like features
        center = (img_size // 2, img_size // 2)
        
        # Face oval
        cv2.ellipse(frame, center, (60, 80), 0, 0, 360, (200, 180, 160), -1)
        
        # Eyes
        cv2.circle(frame, (center[0] - 25, center[1] - 20), 8, (50, 50, 50), -1)
        cv2.circle(frame, (center[0] + 25, center[1] - 20), 8, (50, 50, 50), -1)
        
        # Mouth
        cv2.ellipse(frame, (center[0], center[1] + 30), (20, 10), 0, 0, 180, (100, 50, 50), -1)
        
        # Label: 0 = Real, 1 = Fake
        label = i % 2
        
        if label == 1:
            # Add artifacts for "fake" images
            noise = np.random.randint(-50, 50, frame.shape, dtype=np.int16)
            frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        frames.append(frame)
        labels.append(label)
    
    return frames, labels


if __name__ == "__main__":
    print("Generating synthetic data...")
    frames, labels = generate_synthetic_data(n_samples=1000)
    
    # Split data
    split_idx = int(0.8 * len(frames))
    train_frames, val_frames = frames[:split_idx], frames[split_idx:]
    train_labels, val_labels = labels[:split_idx], labels[split_idx:]
    
    # Create datasets
    train_dataset = FrameLevelDataset(train_frames, train_labels)
    val_dataset = FrameLevelDataset(val_frames, val_labels)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
    
    # Initialize model
    print("\nInitializing model...")
    model = DeepfakeDetector(pretrained=True, num_classes=2)
    
    # Train
    trainer = DeepfakeTrainer(model)
    print("\nStarting training...")
    val_preds, val_labels, val_probs = trainer.train(
        train_loader, val_loader, epochs=20, lr=1e-4
    )
    
    # Plot results
    trainer.plot_training_history()
    trainer.evaluate(val_labels, val_preds, val_probs)
    
    print("\nTraining complete! Model saved as 'best_deepfake_detector.pth'")