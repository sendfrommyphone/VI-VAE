import torch
import torch.nn as nn
import torch.nn.functional as F
import pyro
import pyro.distributions as dist
from torchvision import datasets, transforms
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
import matplotlib.pyplot as plt

# Fix the plot_loss_curve function
def plot_loss_curve(losses, save_path='loss_curve.png'):
    plt.figure(figsize=(10, 6))
    # Convert losses to numpy for plotting if they're tensors
    losses = [loss.cpu().numpy() if torch.is_tensor(loss) else loss for loss in losses]
    plt.plot(losses, 'b-', label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('VAE Training Loss Over Time')
    plt.ylim(-300, 2000)  # Changed y-axis range
    plt.grid(True)
    plt.legend()
    plt.savefig(save_path)
    plt.close()

class Binarize(object):
    def __init__(self, threshold=0.5):
        self.threshold = threshold
    
    def __call__(self, x):
        return (x > self.threshold).float()

class VAE(nn.Module):
    def __init__(self, z_dim=20):
        super(VAE, self).__init__()
        # Encoder with more capacity
        self.encoder = nn.Sequential(
            nn.Linear(784, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(256, z_dim)
        self.fc_logvar = nn.Linear(256, z_dim)
        
        # Decoder with matching architecture
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 784),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        h = self.encoder(x.view(-1, 784))
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return mu + eps*std
        return mu
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

class PyroVAE(pyro.nn.PyroModule):
    def __init__(self, z_dim=20):
        super().__init__()
        self.vae = VAE(z_dim)
        self.z_dim = z_dim
    
    def model(self, x):
        pyro.module("vae", self.vae)
        batch_size = x.shape[0]
        
        with pyro.plate("data", batch_size):
            # Scale the prior
            z_loc = torch.zeros(batch_size, self.z_dim, device=x.device)
            z_scale = torch.ones(batch_size, self.z_dim, device=x.device)
            
            z = pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))
            loc = self.vae.decode(z)
            # Reduce observation noise scale
            scale = torch.ones_like(loc) * 0.1
            
            # Scale likelihood
            pyro.sample(
                "obs",
                dist.Normal(loc, scale).to_event(1),
                obs=x.view(-1, 784)
            )

    def guide(self, x):
        with pyro.plate("data", x.shape[0]):
            # Use neural network to compute parameters
            mu, logvar = self.vae.encode(x.view(-1, 784))
            # Sample from variational distribution
            pyro.sample("latent", dist.Normal(mu, torch.exp(0.5*logvar)).to_event(1))

def visualize_reconstruction(model, data, device, epoch):
    model.eval()
    with torch.no_grad():
        # Get original and reconstruction
        x = data.to(device)
        recon, _, _ = model.vae(x)
        
        # Plot original vs reconstruction
        n = min(8, x.size(0))
        fig, axes = plt.subplots(2, n, figsize=(2*n, 4))
        
        for i in range(n):
            # Original
            axes[0,i].imshow(x[i,0].cpu(), cmap='gray')
            axes[0,i].axis('off')
            if i == 0:
                axes[0,i].set_title('Original')
                
            # Reconstruction
            axes[1,i].imshow(recon[i].view(28,28).cpu(), cmap='gray')
            axes[1,i].axis('off')
            if i == 0:
                axes[1,i].set_title('Reconstructed')
                
        plt.savefig(f'reconstruction_epoch_{epoch+1}.png')
        plt.close()

# Fix the train function
def train(epochs=30, eval_frequency=5):
    train_losses = []
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        num_batches = len(train_loader)
        
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            # Normalize loss by batch size
            loss = svi.step(data) / data.size(0)  # Per-sample loss
            total_loss += loss
            
            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch+1}, Batch: {batch_idx}, Loss per sample: {loss:.4f}')
        
        # Calculate average loss per sample
        avg_loss = total_loss / num_batches
        train_losses.append(avg_loss)
        print(f"Epoch {epoch+1} Average Loss per sample: {avg_loss:.4f}")
        
        # Generate samples and reconstructions periodically
        if (epoch + 1) % eval_frequency == 0:
            model.eval()
            with torch.no_grad():
                # Get a batch of validation data
                val_data, _ = next(iter(train_loader))
                visualize_reconstruction(model, val_data, device, epoch)
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'loss': best_loss,
            }, 'best_vae_model.pt')
    
    # Plot final loss curve
    plot_loss_curve(train_losses, save_path='final_loss_curve.png')
    return train_losses

# Add this function after the train function
def load_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Loaded model from epoch {epoch} with loss {loss:.4f}")
    return model

# Data loading with augmentation
transform = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    Binarize(threshold=0.5)  # Replace Lambda with custom class
])

# Load and prepare data
train_data = datasets.MNIST('./mnist_data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(
    train_data, 
    batch_size=64,
    shuffle=True,
    num_workers=0,  # Set to 0 if still having issues
    pin_memory=True if torch.cuda.is_available() else False
)

# Update the model configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pyro.clear_param_store()
model = PyroVAE(z_dim=20).to(device)
optimizer = Adam({
    "lr": 1e-4,
    "betas": (0.9, 0.999),
    "weight_decay": 1e-5
})
svi = SVI(
    model=model.model,
    guide=model.guide,
    optim=optimizer,
    loss=Trace_ELBO(num_particles=1)
)

if __name__ == '__main__':
    # Windows requires this for multiprocessing
    import multiprocessing
    multiprocessing.freeze_support()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    losses = train(epochs=30, eval_frequency=5)