"""
Image Super-Resolution using SRGAN (Super-Resolution GAN)
Upscales low-resolution images to high-resolution
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

# Install: pip install torch torchvision pillow


class ResidualBlock(nn.Module):
    """Residual block for generator"""
    
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        residual = x
        out = self.prelu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return out + residual


class UpsampleBlock(nn.Module):
    """Upsample block using sub-pixel convolution"""
    
    def __init__(self, in_channels, upscale_factor):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * (upscale_factor ** 2), 
                             kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        self.prelu = nn.PReLU()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x


class Generator(nn.Module):
    """SRGAN Generator network"""
    
    def __init__(self, scale_factor=4, num_residual_blocks=16):
        super().__init__()
        self.scale_factor = scale_factor
        
        # Initial convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )
        
        # Residual blocks
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(64) for _ in range(num_residual_blocks)]
        )
        
        # Post-residual convolution
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64)
        )
        
        # Upsampling blocks
        upsample_blocks = []
        for _ in range(scale_factor // 2):
            upsample_blocks.append(UpsampleBlock(64, 2))
        self.upsample = nn.Sequential(*upsample_blocks)
        
        # Final output convolution
        self.conv3 = nn.Conv2d(64, 3, kernel_size=9, padding=4)
    
    def forward(self, x):
        conv1 = self.conv1(x)
        residual = self.residual_blocks(conv1)
        conv2 = self.conv2(residual)
        out = conv1 + conv2
        out = self.upsample(out)
        out = self.conv3(out)
        return torch.tanh(out)


class Discriminator(nn.Module):
    """SRGAN Discriminator network"""
    
    def __init__(self):
        super().__init__()
        
        def discriminator_block(in_channels, out_channels, stride=1, bn=True):
            layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1)]
            if bn:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)
        
        self.features = nn.Sequential(
            discriminator_block(3, 64, bn=False),
            discriminator_block(64, 64, stride=2),
            discriminator_block(64, 128),
            discriminator_block(128, 128, stride=2),
            discriminator_block(128, 256),
            discriminator_block(256, 256, stride=2),
            discriminator_block(256, 512),
            discriminator_block(512, 512, stride=2)
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        features = self.features(x)
        out = self.classifier(features)
        return out.view(out.size(0), -1)


class VGGPerceptualLoss(nn.Module):
    """Perceptual loss using VGG19 features"""
    
    def __init__(self):
        super().__init__()
        vgg = models.vgg19(pretrained=True).features
        self.blocks = nn.ModuleList([
            vgg[:4].eval(),    # relu1_2
            vgg[4:9].eval(),   # relu2_2
            vgg[9:18].eval(),  # relu3_4
            vgg[18:27].eval(), # relu4_4
            vgg[27:36].eval()  # relu5_4
        ])
        
        # Freeze parameters
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, x, y):
        loss = 0
        for block in self.blocks:
            x = block(x)
            y = block(y)
            loss += F.mse_loss(x, y)
        return loss


class ImageDataset(Dataset):
    """Dataset for super-resolution training"""
    
    def __init__(self, hr_images, scale_factor=4, patch_size=96):
        self.hr_images = hr_images
        self.scale_factor = scale_factor
        self.patch_size = patch_size
        self.lr_patch_size = patch_size // scale_factor
        
        self.hr_transform = transforms.Compose([
            transforms.RandomCrop(patch_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        self.lr_transform = transforms.Compose([
            transforms.Resize(self.lr_patch_size, Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    def __len__(self):
        return len(self.hr_images)
    
    def __getitem__(self, idx):
        hr_image = self.hr_images[idx]
        
        # Apply transform
        hr_image = self.hr_transform(hr_image)
        
        # Create LR image
        lr_image = F.interpolate(
            hr_image.unsqueeze(0), 
            scale_factor=1/self.scale_factor,
            mode='bicubic',
            align_corners=False
        ).squeeze(0)
        
        return lr_image, hr_image


class SRGANTrainer:
    """Training pipeline for SRGAN"""
    
    def __init__(self, generator, discriminator, 
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.device = device
        
        self.perceptual_loss = VGGPerceptualLoss().to(device)
        self.adversarial_loss = nn.BCELoss()
        self.content_loss = nn.MSELoss()
        
        self.g_losses = []
        self.d_losses = []
        self.psnr_values = []
    
    def calculate_psnr(self, img1, img2):
        """Calculate PSNR between two images"""
        mse = torch.mean((img1 - img2) ** 2)
        if mse == 0:
            return 100
        return 20 * torch.log10(1.0 / torch.sqrt(mse))
    
    def train_epoch(self, train_loader, g_optimizer, d_optimizer, epoch):
        """Train for one epoch"""
        self.generator.train()
        self.discriminator.train()
        
        g_loss_epoch = 0
        d_loss_epoch = 0
        psnr_epoch = 0
        
        for lr_imgs, hr_imgs in tqdm(train_loader, desc=f"Epoch {epoch}"):
            batch_size = lr_imgs.size(0)
            lr_imgs = lr_imgs.to(self.device)
            hr_imgs = hr_imgs.to(self.device)
            
            # Real and fake labels
            real_labels = torch.ones(batch_size, 1).to(self.device)
            fake_labels = torch.zeros(batch_size, 1).to(self.device)
            
            # ==================== Train Discriminator ====================
            d_optimizer.zero_grad()
            
            # Real images
            real_output = self.discriminator(hr_imgs)
            d_real_loss = self.adversarial_loss(real_output, real_labels)
            
            # Fake images
            fake_imgs = self.generator(lr_imgs)
            fake_output = self.discriminator(fake_imgs.detach())
            d_fake_loss = self.adversarial_loss(fake_output, fake_labels)
            
            # Total discriminator loss
            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            d_optimizer.step()
            
            # ==================== Train Generator ====================
            g_optimizer.zero_grad()
            
            # Generate fake images
            fake_imgs = self.generator(lr_imgs)
            fake_output = self.discriminator(fake_imgs)
            
            # Adversarial loss
            adversarial_loss = self.adversarial_loss(fake_output, real_labels)
            
            # Perceptual loss
            perceptual_loss = self.perceptual_loss(fake_imgs, hr_imgs)
            
            # Content loss (MSE)
            content_loss = self.content_loss(fake_imgs, hr_imgs)
            
            # Total generator loss
            g_loss = content_loss + 0.001 * adversarial_loss + 0.006 * perceptual_loss
            g_loss.backward()
            g_optimizer.step()
            
            # Calculate PSNR
            psnr = self.calculate_psnr(fake_imgs, hr_imgs)
            
            # Accumulate losses
            g_loss_epoch += g_loss.item()
            d_loss_epoch += d_loss.item()
            psnr_epoch += psnr.item()
        
        n = len(train_loader)
        return g_loss_epoch / n, d_loss_epoch / n, psnr_epoch / n
    
    def train(self, train_loader, epochs=100, g_lr=1e-4, d_lr=1e-4):
        """Complete training loop"""
        g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=g_lr)
        d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=d_lr)
        
        g_scheduler = torch.optim.lr_scheduler.StepLR(g_optimizer, step_size=50, gamma=0.5)
        d_scheduler = torch.optim.lr_scheduler.StepLR(d_optimizer, step_size=50, gamma=0.5)
        
        best_psnr = 0
        
        for epoch in range(1, epochs + 1):
            g_loss, d_loss, psnr = self.train_epoch(
                train_loader, g_optimizer, d_optimizer, epoch
            )
            
            # Update schedulers
            g_scheduler.step()
            d_scheduler.step()
            
            # Save metrics
            self.g_losses.append(g_loss)
            self.d_losses.append(d_loss)
            self.psnr_values.append(psnr)
            
            print(f"\nEpoch {epoch}/{epochs}")
            print(f"G Loss: {g_loss:.4f}, D Loss: {d_loss:.4f}, PSNR: {psnr:.2f} dB")
            
            # Save best model
            if psnr > best_psnr:
                best_psnr = psnr
                torch.save(self.generator.state_dict(), 'best_generator.pth')
                torch.save(self.discriminator.state_dict(), 'best_discriminator.pth')
                print(f"Saved best model with PSNR: {best_psnr:.2f} dB")
    
    def plot_training_history(self):
        """Plot training curves"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Generator loss
        axes[0].plot(self.g_losses, label='Generator Loss', color='blue')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Generator Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Discriminator loss
        axes[1].plot(self.d_losses, label='Discriminator Loss', color='red')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('Discriminator Loss')
        axes[1].legend()
        axes[1].grid(True)
        
        # PSNR
        axes[2].plot(self.psnr_values, label='PSNR', color='green')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('PSNR (dB)')
        axes[2].set_title('Peak Signal-to-Noise Ratio')
        axes[2].legend()
        axes[2].grid(True)
        
        plt.tight_layout()
        plt.savefig('srgan_training_history.png', dpi=300)
        plt.show()
    
    def upscale_image(self, lr_image):
        """Upscale a single low-resolution image"""
        self.generator.eval()
        
        with torch.no_grad():
            # Prepare image
            if isinstance(lr_image, np.ndarray):
                lr_image = Image.fromarray(lr_image)
            
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
            
            lr_tensor = transform(lr_image).unsqueeze(0).to(self.device)
            
            # Generate SR image
            sr_tensor = self.generator(lr_tensor)
            
            # Denormalize
            sr_tensor = sr_tensor * 0.5 + 0.5
            sr_tensor = torch.clamp(sr_tensor, 0, 1)
            
            # Convert to numpy
            sr_image = sr_tensor.cpu().squeeze(0).permute(1, 2, 0).numpy()
            sr_image = (sr_image * 255).astype(np.uint8)
            
        return sr_image
    
    def visualize_results(self, test_images, num_samples=4):
        """Visualize super-resolution results"""
        fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4*num_samples))
        
        for i in range(num_samples):
            lr_image = test_images[i]
            
            # Create bicubic upsampled version
            h, w = lr_image.size
            bicubic = lr_image.resize((w*4, h*4), Image.BICUBIC)
            
            # Generate SR image
            sr_image = self.upscale_image(lr_image)
            
            # Plot
            axes[i, 0].imshow(lr_image)
            axes[i, 0].set_title('Low Resolution')
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(bicubic)
            axes[i, 1].set_title('Bicubic Upsampling')
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(sr_image)
            axes[i, 2].set_title('SRGAN (Our Method)')
            axes[i, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig('super_resolution_results.png', dpi=300)
        plt.show()


# Generate synthetic data
def generate_synthetic_images(n_samples=500, size=256):
    """Generate synthetic high-resolution images"""
    images = []
    
    for _ in range(n_samples):
        # Create random colored patterns
        img = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
        
        # Add some structure (circles, rectangles)
        for _ in range(5):
            center = (np.random.randint(0, size), np.random.randint(0, size))
            radius = np.random.randint(10, 50)
            color = tuple(np.random.randint(0, 255, 3).tolist())
            img = cv2.circle(img, center, radius, color, -1)
        
        images.append(Image.fromarray(img))
    
    return images


if __name__ == "__main__":
    print("Generating synthetic high-resolution images...")
    hr_images = generate_synthetic_images(n_samples=200, size=256)
    
    # Split data
    train_images = hr_images[:160]
    test_images = hr_images[160:]
    
    # Create dataset and dataloader
    train_dataset = ImageDataset(train_images, scale_factor=4, patch_size=96)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
    
    # Initialize models
    print("\nInitializing SRGAN models...")
    generator = Generator(scale_factor=4, num_residual_blocks=16)
    discriminator = Discriminator()
    
    # Train
    trainer = SRGANTrainer(generator, discriminator)
    print("\nStarting SRGAN training...")
    trainer.train(train_loader, epochs=50, g_lr=1e-4, d_lr=1e-4)
    
    # Plot results
    trainer.plot_training_history()
    
    # Create low-resolution test images
    lr_test_images = [img.resize((64, 64), Image.BICUBIC) for img in test_images[:4]]
    trainer.visualize_results(lr_test_images, num_samples=4)
    
    print("\nTraining complete! Models saved.")