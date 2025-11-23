import torch
import torch.nn as nn

# ============================================================
# 1. FCNN
# ============================================================

class FCNN(nn.Module):
    def __init__(self, in_dim=32*32*3, hidden=512, num_classes=10, p=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True),
            nn.Dropout(p),
            nn.Linear(hidden, 128), nn.ReLU(inplace=True),
            nn.Dropout(p),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        return self.net(x)


# ============================================================
# 2. Simple CNN
# ============================================================

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10, p=0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p),
            nn.Linear(64*8*8, 128), nn.ReLU(inplace=True),
            nn.Dropout(p),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ============================================================
# 3. Enhanced CNN
# ============================================================

class EnhancedCNN(nn.Module):
    def __init__(self, num_classes=10, p=0.4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p),
            nn.Linear(128*8*8, 256), nn.ReLU(inplace=True),
            nn.Dropout(p),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ============================================================
# 4. VAE
# ============================================================

class VAE(nn.Module):
    def __init__(self, z_dim=64):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(True)
        )
        self.enc_flat = nn.Flatten()
        self.fc_mu = nn.Linear(128*4*4, z_dim)
        self.fc_logvar = nn.Linear(128*4*4, z_dim)

        self.fc_dec = nn.Linear(z_dim, 128*4*4)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Sigmoid()
        )

    def encode(self, x):
        h = self.enc_flat(self.enc(x))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_dec(z).view(-1, 128, 4, 4)
        return self.dec(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


# ============================================================
# 5. EBM
# ============================================================

class SimpleEBM(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256*4*4, 1)
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# 6. Diffusion (TinyUNet)
# ============================================================

from .diffusion_generator import TinyUNet


# ============================================================
# 7. GAN
# ============================================================

class GAN(nn.Module):
    def __init__(self):
        super().__init__()
        from .generator import Generator
        self.generator = Generator()

    def forward(self, z):
        return self.generator(z)


# ============================================================
# 8. get_model()
# ============================================================

def get_model(name: str):
    name = name.lower()

    if name == "cnn": return SimpleCNN()
    if name == "enhancedcnn": return EnhancedCNN()
    if name == "fcnn": return FCNN()
    if name == "vae": return VAE()
    if name == "gan": return GAN()
    if name == "diffusion": return TinyUNet()
    if name == "ebm": return SimpleEBM()

    raise ValueError(f"Unknown model: {name}")