"""
Facial Expression Recognition with Multi-Task Learning
Recognizes 7 emotions: angry, disgust, fear, happy, sad, surprise, neutral
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from tqdm import tqdm

# Install: pip install torch torchvision opencv-python scikit-learn seaborn


class SpatialAttention(nn.Module):
    """Spatial attention for facial regions"""
    
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(concat))
        return x * attention


class EmotionRecognitionModel(nn.Module):
    """CNN with attention for emotion recognition"""
    
    def __init__(self, num_classes=7, pretrained=True):
        super().__init__()
        
        # Use ResNet18 as backbone
        resnet = models.resnet18(pretrained=pretrained)
        
        # Remove final FC layer
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        
        # Spatial attention
        self.spatial_attention = SpatialAttention()
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Emotion classifier
        self.emotion_classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        # Auxiliary task: Arousal-Valence regression
        self.arousal_regressor = nn.Linear(512, 1)
        self.valence_regressor = nn.Linear(512, 1)
    
    def forward(self, x, return_features=False):
        # Extract features
        features = self.features(x)
        
        # Apply spatial attention
        attended = self.spatial_attention(features)
        
        # Global pooling
        pooled = self.global_pool(attended)
        pooled = pooled.view(pooled.size(0), -1)
        
        # Emotion classification
        emotion_logits = self.emotion_classifier(pooled)
        
        # Arousal-Valence prediction
        arousal = self.arousal_regressor(pooled)
        valence = self.valence_regressor(pooled)
        
        if return_features:
            return emotion_logits, arousal, valence, pooled
        
        return emotion_logits, arousal, valence


class FacialExpressionDataset(Dataset):
    """Dataset for facial expression images"""
    
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
        
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


class EmotionRecognitionTrainer:
    """Training pipeline for emotion recognition"""
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.emotion_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        
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
        
        for images, labels in tqdm(train_loader, desc="Training"):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            optimizer.zero_grad()
            
            # Forward pass
            emotion_logits, arousal, valence = self.model(images)
            
            # Emotion classification loss
            emotion_loss = criterion(emotion_logits, labels)
            
            # Total loss (can add arousal/valence loss if labels available)
            loss = emotion_loss
            
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = emotion_logits.max(1)
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
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validation"):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                emotion_logits, _, _ = self.model(images)
                loss = criterion(emotion_logits, labels)
                
                running_loss += loss.item()
                _, predicted = emotion_logits.max(1)
                
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        epoch_loss = running_loss / len(val_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc, all_preds, all_labels
    
    def train(self, train_loader, val_loader, epochs=50, lr=0.001):
        """Complete training loop"""
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', patience=5, factor=0.5
        )
        
        best_acc = 0
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, criterion)
            
            # Validate
            val_loss, val_acc, val_preds, val_labels = self.validate(val_loader, criterion)
            
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
                torch.save(self.model.state_dict(), 'best_emotion_model.pth')
                print(f"Saved best model with accuracy: {best_acc:.2f}%")
        
        return val_preds, val_labels
    
    def predict_emotion(self, image):
        """Predict emotion from single image"""
        self.model.eval()
        
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        image_tensor = transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            emotion_logits, arousal, valence = self.model(image_tensor)
            probs = F.softmax(emotion_logits, dim=1)
            pred_class = emotion_logits.argmax(1).item()
            confidence = probs[0, pred_class].item()
        
        return {
            'emotion': self.emotion_names[pred_class],
            'confidence': confidence,
            'arousal': arousal.item(),
            'valence': valence.item(),
            'all_probs': {name: prob.item() for name, prob in zip(self.emotion_names, probs[0])}
        }
    
    def predict_from_webcam(self, duration=30):
        """Real-time emotion recognition from webcam"""
        cap = cv2.VideoCapture(0)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        start_time = cv2.getTickCount()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect faces
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            for (x, y, w, h) in faces:
                # Extract face
                face = frame[y:y+h, x:x+w]
                face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                
                # Predict emotion
                result = self.predict_emotion(face_rgb)
                
                # Draw rectangle and text
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                text = f"{result['emotion']}: {result['confidence']:.2f}"
                cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.9, (0, 255, 0), 2)
            
            cv2.imshow('Emotion Recognition', frame)
            
            # Check duration
            elapsed = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
            if elapsed > duration or cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    def plot_training_history(self):
        """Plot training curves"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss
        axes[0].plot(self.train_losses, label='Train Loss', marker='o')
        axes[0].plot(self.val_losses, label='Val Loss', marker='s')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training History - Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Accuracy
        axes[1].plot(self.train_accs, label='Train Acc', marker='o')
        axes[1].plot(self.val_accs, label='Val Acc', marker='s')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title('Training History - Accuracy')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig('emotion_training_history.png', dpi=300)
        plt.show()
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.emotion_names,
                   yticklabels=self.emotion_names)
        plt.title('Confusion Matrix - Emotion Recognition')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('emotion_confusion_matrix.png', dpi=300)
        plt.show()
    
    def evaluate(self, y_true, y_pred):
        """Comprehensive evaluation"""
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, 
                                   target_names=self.emotion_names,
                                   digits=4))
        
        self.plot_confusion_matrix(y_true, y_pred)
    
    def visualize_predictions(self, images, labels, num_samples=8):
        """Visualize predictions on sample images"""
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.ravel()
        
        for i in range(min(num_samples, len(images))):
            result = self.predict_emotion(images[i])
            
            axes[i].imshow(images[i])
            axes[i].set_title(f"True: {self.emotion_names[labels[i]]}\n"
                            f"Pred: {result['emotion']} ({result['confidence']:.2f})")
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig('emotion_predictions.png', dpi=300)
        plt.show()


# Generate synthetic facial expression data
def generate_synthetic_faces(n_samples=5000, img_size=48):
    """Generate synthetic face-like images with emotions"""
    images = []
    labels = []
    
    for _ in range(n_samples):
        # Create base face
        img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 200
        
        # Face shape (circle)
        center = (img_size // 2, img_size // 2)
        radius = img_size // 3
        cv2.circle(img, center, radius, (220, 200, 180), -1)
        
        # Emotion (0-6)
        emotion = np.random.randint(0, 7)
        
        # Eyes
        left_eye = (center[0] - radius//3, center[1] - radius//4)
        right_eye = (center[0] + radius//3, center[1] - radius//4)
        
        if emotion == 0:  # Angry
            cv2.line(img, (left_eye[0]-5, left_eye[1]-3), (left_eye[0]+5, left_eye[1]), (0, 0, 0), 2)
            cv2.line(img, (right_eye[0]-5, right_eye[1]), (right_eye[0]+5, right_eye[1]-3), (0, 0, 0), 2)
            cv2.ellipse(img, (center[0], center[1]+radius//3), (radius//3, radius//6), 
                       0, 0, 180, (0, 0, 0), 2)
        elif emotion == 3:  # Happy
            cv2.circle(img, left_eye, 3, (0, 0, 0), -1)
            cv2.circle(img, right_eye, 3, (0, 0, 0), -1)
            cv2.ellipse(img, (center[0], center[1]+radius//4), (radius//3, radius//4), 
                       0, 180, 360, (0, 0, 0), 2)
        elif emotion == 4:  # Sad
            cv2.circle(img, left_eye, 3, (0, 0, 0), -1)
            cv2.circle(img, right_eye, 3, (0, 0, 0), -1)
            cv2.ellipse(img, (center[0], center[1]+radius//3), (radius//3, radius//6), 
                       0, 0, 180, (0, 0, 0), 2)
        else:  # Others
            cv2.circle(img, left_eye, 3, (0, 0, 0), -1)
            cv2.circle(img, right_eye, 3, (0, 0, 0), -1)
            cv2.line(img, (center[0]-radius//4, center[1]+radius//3), 
                    (center[0]+radius//4, center[1]+radius//3), (0, 0, 0), 2)
        
        # Add noise
        noise = np.random.randint(-20, 20, img.shape, dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        images.append(img)
        labels.append(emotion)
    
    return images, labels


if __name__ == "__main__":
    print("Generating synthetic facial expression data...")
    images, labels = generate_synthetic_faces(n_samples=3000, img_size=48)
    
    # Split data
    split_idx = int(0.8 * len(images))
    train_images, val_images = images[:split_idx], images[split_idx:]
    train_labels, val_labels = labels[:split_idx], labels[split_idx:]
    
    # Create datasets
    train_dataset = FacialExpressionDataset(train_images, train_labels)
    val_dataset = FacialExpressionDataset(val_images, val_labels)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
    
    # Initialize model
    print("\nInitializing Emotion Recognition model...")
    model = EmotionRecognitionModel(num_classes=7, pretrained=True)
    
    # Train
    trainer = EmotionRecognitionTrainer(model)
    print("\nStarting training...")
    val_preds, val_labels = trainer.train(train_loader, val_loader, epochs=30, lr=0.001)
    
    # Plot results
    trainer.plot_training_history()
    trainer.evaluate(val_labels, val_preds)
    trainer.visualize_predictions(val_images[:8], val_labels[:8])
    
    print("\nTraining complete! Model saved as 'best_emotion_model.pth'")
    print("\nTo use webcam: trainer.predict_from_webcam(duration=30)")