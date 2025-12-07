import math
import torch
import torch.nn as nn



class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb



class TinyUNet(nn.Module):
    def __init__(self, time_emb_dim=128):
        super().__init__()

        # time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, 64),
            nn.ReLU()
        )

        # encoder
        self.down1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU()
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU()
        )

        # decoder
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, padding=1),
            nn.ReLU()
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(64, 3, 3, padding=1)
        )

    def forward(self, x, t):
        # time embedding -> add to feature maps
        t_emb = self.time_mlp(t).unsqueeze(-1).unsqueeze(-1)

        d1 = self.down1(x)
        d2 = self.down2(d1 + t_emb)

        u1 = self.up1(d2)
        out = self.up2(u1 + d1)

        return out


def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / steps) + s) / (1 + s) * math.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


class DiffusionGenerator:
    """
    DDPM sampling using Tiny-UNet.
    """

    def __init__(self, model: nn.Module, device="cpu", num_steps=200):
        self.model = model.to(device)
        self.device = device
        self.num_steps = num_steps

        self.betas = cosine_beta_schedule(num_steps).to(device)
        self.alphas = 1. - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    @torch.no_grad()
    def sample(self, n=16):
        x = torch.randn(n, 3, 32, 32, device=self.device)

        for t in reversed(range(self.num_steps)):
            t_batch = torch.full((n,), t, device=self.device, dtype=torch.long)

            alpha_t = self.alphas[t]
            alpha_bar_t = self.alpha_bars[t]
            beta_t = self.betas[t]

            eps_theta = self.model(x, t_batch)

            x = (1.0 / torch.sqrt(alpha_t)) * (
                x - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * eps_theta
            )

            if t > 0:
                z = torch.randn_like(x)
                x = x + torch.sqrt(beta_t) * z

        return x.clamp(0.0, 1.0)