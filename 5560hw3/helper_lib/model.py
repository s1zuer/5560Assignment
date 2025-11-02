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
    def forward(self, x):
        return self.net(x)

# ----- 基础 CNN（两层卷积 + 池化） -----
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10, p=0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32->16

            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16->8
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
            nn.MaxPool2d(2),                 # 32->16

            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                 # 16->8
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

# ===== VAE（Module 5：Encoder -> μ, logσ²；Decoder；KL+重建） =====
class VAE(nn.Module):
    def __init__(self, z_dim=64):
        super().__init__()
        # Encoder: 把 32x32x3 压成均值/方差
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.ReLU(True),   # -> 16x16
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(True),  # -> 8x8
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(True)  # -> 4x4
        )
        self.enc_flat = nn.Flatten()
        self.fc_mu = nn.Linear(128*4*4, z_dim)
        self.fc_logvar = nn.Linear(128*4*4, z_dim)

        # Decoder: 从潜在向量还原成图像
        self.fc_dec = nn.Linear(z_dim, 128*4*4)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(True),   # -> 8x8
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(True),    # -> 16x16
            nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Sigmoid()       # -> 32x32, [0,1]
        )

    def encode(self, x):
        h = self.enc_flat(self.enc(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        # 采样 z ~ N(mu, sigma^2) 使用 reparam trick
        std = (0.5 * logvar).exp()
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





# ===== 你的前面那些 FCNN / SimpleCNN / EnhancedCNN / VAE 保持不动 =====
# 这里只重写 GAN 相关部分和 get_model

class Generator(nn.Module):
    def __init__(self, z_dim=100, img_channels=1, feature_maps=64):
        super().__init__()
        self.z_dim = z_dim

        # 1) 先用全连接把随机噪声拉成一个 [128, 7, 7] 的feature map
        self.fc = nn.Sequential(
            nn.Linear(z_dim, 128 * 7 * 7),
            nn.BatchNorm1d(128 * 7 * 7),
            nn.ReLU(True)
        )

        # 2) 然后用反卷积两次：7x7 -> 14x14 -> 28x28
        self.deconv = nn.Sequential(
            # [N,128,7,7] -> [N,64,14,14]
            nn.ConvTranspose2d(
                in_channels=128,
                out_channels=feature_maps,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(feature_maps),
            nn.ReLU(True),

            # [N,64,14,14] -> [N,1,28,28]
            nn.ConvTranspose2d(
                in_channels=feature_maps,
                out_channels=img_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.Tanh()  # 输出范围 [-1,1]，正好对应我们对MNIST做的Normalize((0.5,), (0.5,))
        )

    def forward(self, z):
        # 输入 z: [batch, z_dim]
        x = self.fc(z)                        # [batch, 128*7*7]
        x = x.view(-1, 128, 7, 7)             # [batch,128,7,7]
        img = self.deconv(x)                  # [batch,1,28,28]
        return img


class Discriminator(nn.Module):
    def __init__(self, img_channels=1, feature_maps=64):
        super().__init__()

        # 下采样两次：28x28 -> 14x14 -> 7x7
        self.conv = nn.Sequential(
            # [N,1,28,28] -> [N,64,14,14]
            nn.Conv2d(
                in_channels=img_channels,
                out_channels=feature_maps,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.LeakyReLU(0.2, inplace=True),

            # [N,64,14,14] -> [N,128,7,7]
            nn.Conv2d(
                in_channels=feature_maps,
                out_channels=feature_maps * 2,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # 判定层：把 [128,7,7] 拉平到 128*7*7，然后线性到1，再Sigmoid
        self.classifier = nn.Sequential(
            nn.Linear((feature_maps * 2) * 7 * 7, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        h = self.conv(x)                      # [batch,128,7,7]  (因为 feature_maps=64 → feature_maps*2=128)
        h = h.view(h.size(0), -1)             # [batch, 128*7*7]
        out = self.classifier(h)              # [batch,1]
        return out


class GAN(nn.Module):
    def __init__(self, z_dim=100):
        super().__init__()
        self.z_dim = z_dim
        self.generator = Generator(z_dim=z_dim, img_channels=1, feature_maps=64)
        self.discriminator = Discriminator(img_channels=1, feature_maps=64)

    def forward(self, z):
        return self.generator(z)


# ===== 工厂方法：更新成能返回GAN =====
def get_model(model_name: str):
    """
    根据字符串返回需要的模型实例
    支持: FCNN, CNN, EnhancedCNN, VAE, GAN
    """
    name = model_name.lower()
    if name == "fcnn":
        return FCNN()
    if name == "cnn":
        return SimpleCNN()
    if name == "enhancedcnn":
        return EnhancedCNN()
    if name == "vae":
        return VAE()
    if name == "gan":
        return GAN()
    raise ValueError(f"Unknown model_name: {model_name}")