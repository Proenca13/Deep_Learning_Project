import torch
import torch.nn as nn

LATENT_DIM = 100

def spectral_norm(module):
    """Convenience wrapper for nn.utils.spectral_norm."""
    return nn.utils.spectral_norm(module)

class Generator_v3(nn.Module):
    """
    WGAN-GP Generator: latent(100) → 128×128 RGB image.
    Uses large capacity (ngf=128) and BatchNorm.
    """
    def __init__(self, latent_dim=LATENT_DIM, ngf=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, ngf * 8, 8, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8), nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4), nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2), nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf), nn.ReLU(True),

            nn.ConvTranspose2d(ngf, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        if z.dim() == 2:
            z = z.view(z.size(0), z.size(1), 1, 1)
        return self.net(z)

class Critic_v3(nn.Module):
    """
    WGAN-GP Critic: 128x128 RGB -> scalar score.
    
    Key differences:
    - Uses InstanceNorm2d instead of BatchNorm (crucial for Gradient Penalty).
    - No sigmoid activation at the end (outputs raw scores).
    """
    def __init__(self, ndf=64):
        super().__init__()
        self.net = nn.Sequential(
            # Block 1 — no norm on first layer
            spectral_norm(nn.Conv2d(3, ndf, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),

            # Block 2
            spectral_norm(nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)),
            nn.InstanceNorm2d(ndf * 2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # Block 3
            spectral_norm(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)),
            nn.InstanceNorm2d(ndf * 4, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # Block 4
            spectral_norm(nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False)),
            nn.InstanceNorm2d(ndf * 8, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # Output — raw score
            spectral_norm(nn.Conv2d(ndf * 8, 1, 8, 1, 0, bias=False)),
        )

    def forward(self, x):
        return self.net(x).view(-1)