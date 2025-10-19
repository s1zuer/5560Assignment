import torch
import torch.nn as nn

# ----- FCNN（用于平铺后的小型全连接演示） -----
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
    def forward(self, x): return self.net(x)

# ----- 基础 CNN（两层卷积 + 池化） -----
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

# ----- 加强版 CNN（多一层 / 更大通道） -----
class EnhancedCNN(nn.Module):
    def __init__(self, num_classes=10, p=0.4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                 # 16x16

            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                 # 8x8
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

# ----- VAE（Module5：Encoder->μ,logσ²；Decoder；KL+重建） -----
class VAE(nn.Module):
    def __init__(self, z_dim=64):
        super().__init__()
        # Encoder
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.ReLU(True),  # 16x16
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(True), # 8x8
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(True) # 4x4
        )
        self.enc_flat = nn.Flatten()
        self.fc_mu = nn.Linear(128*4*4, z_dim)
        self.fc_logvar = nn.Linear(128*4*4, z_dim)
        # Decoder
        self.fc_dec = nn.Linear(z_dim, 128*4*4)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(True),  # 8x8
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(True),   # 16x16
            nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Sigmoid()      # 32x32, [0,1]
        )

    def encode(self, x):
        h = self.enc_flat(self.enc(x))
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = (0.5*logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_dec(z).view(-1, 128, 4, 4)
        return self.dec(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar

def get_model(model_name: str):
    name = model_name.lower()
    if name == "fcnn":
        return FCNN()
    if name == "cnn":
        return SimpleCNN()
    if name == "enhancedcnn":
        return EnhancedCNN()
    if name == "vae":
        return VAE()
    raise ValueError(f"Unknown model_name: {model_name}")