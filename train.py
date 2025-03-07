import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from tqdm import tqdm

from model import Generator, Discriminator
from data_loader import get_celeba_loader

def train_dcgan(data_path='./data', output_dir='./output', num_epochs=10,
               batch_size=128, latent_dim=100, lr=0.0002, beta1=0.5,
               save_interval=2, device=None):
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize models
    netG = Generator(latent_dim=latent_dim).to(device)
    netD = Discriminator().to(device)
    
    # Initialize weights
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
    
    netG.apply(weights_init)
    netD.apply(weights_init)
    
    # Setup optimizers
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
    
    # Loss function
    criterion = nn.BCELoss()
    
    # Get data loader
    dataloader = get_celeba_loader(data_path, batch_size)
    
    # Training loop
    g_losses = []
    d_losses = []
    fixed_noise = torch.randn(64, latent_dim, 1, 1, device=device)
    
    print("Starting Training...")
    for epoch in range(num_epochs):
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
        for i, data in progress_bar:
            ############################
            # Update Discriminator
            ###########################
            netD.zero_grad()
            real = data[0].to(device)
            batch_size = real.size(0)
            label_real = torch.ones(batch_size, device=device)
            label_fake = torch.zeros(batch_size, device=device)
            
            # Train with real
            output = netD(real)
            errD_real = criterion(output, label_real)
            D_x = output.mean().item()
            
            # Train with fake
            noise = torch.randn(batch_size, latent_dim, 1, 1, device=device)
            fake = netG(noise)
            output = netD(fake.detach())
            errD_fake = criterion(output, label_fake)
            D_G_z1 = output.mean().item()
            
            errD = errD_real + errD_fake
            errD.backward()
            optimizerD.step()
            
            ############################
            # Update Generator
            ###########################
            netG.zero_grad()
            output = netD(fake)
            errG = criterion(output, label_real)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()
            
            # Update progress bar
            progress_bar.set_description(f'Epoch [{epoch+1}/{num_epochs}]')
            progress_bar.set_postfix(
                **{
                    'D_loss': f'{errD.item():.4f}',
                    'G_loss': f'{errG.item():.4f}',
                    'D(x)': f'{D_x:.4f}',
                    'D(G(z))': f'{D_G_z2:.4f}'
                }
            )
            
            # Save losses for plotting
            g_losses.append(errG.item())
            d_losses.append(errD.item())
        
        # Save images
        if (epoch + 1) % save_interval == 0:
            with torch.no_grad():
                fake = netG(fixed_noise).cpu()
                plt.figure(figsize=(8, 8))
                plt.axis('off')
                plt.title(f'Fake Images (Epoch {epoch+1})')
                plt.imshow(
                    np.transpose(
                        vutils.make_grid(fake[:64], padding=2, normalize=True),
                        (1, 2, 0)
                    )
                )
                plt.savefig(os.path.join(output_dir, f'fake_samples_epoch_{epoch+1}.png'))
                plt.close()
            
            # Save models
            torch.save(netG.state_dict(),
                      os.path.join(output_dir, f'netG_epoch_{epoch+1}.pth'))
            torch.save(netD.state_dict(),
                      os.path.join(output_dir, f'netD_epoch_{epoch+1}.pth'))
    
    # Plot losses
    plt.figure(figsize=(10, 5))
    plt.title('Generator and Discriminator Loss')
    plt.plot(g_losses, label='Generator')
    plt.plot(d_losses, label='Discriminator')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'loss_plot.png'))
    plt.close()
    
    return netG, netD

if __name__ == '__main__':
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Train the model
    train_dcgan()