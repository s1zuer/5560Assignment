import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

from helper_lib.model import get_model     # 会拿到 TinyUNet()
import numpy as np
import os

DEVICE = "cuda" if torch.cuda.is_available() else \
         "mps" if torch.backends.mps.is_available() else "cpu"

# ----------------------------------------------------
# Diffusion 超参数
# ----------------------------------------------------
T = 200     # Diffusion steps（前向/反向）
EPOCHS = 50
LR = 1e-4
BATCH = 64


# ----------------------------------------------------
# Beta schedule（Variance schedule）
# ----------------------------------------------------
def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / steps) + s) / (1 + s) * torch.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


betas = cosine_beta_schedule(T).to(DEVICE)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)


# ----------------------------------------------------
# 前向扩散公式：得到 x_t
# ----------------------------------------------------
def q_sample(x0, t, noise):
    sqrt_acp = torch.sqrt(alphas_cumprod[t]).view(-1, 1, 1, 1)
    sqrt_m1 = torch.sqrt(1 - alphas_cumprod[t]).view(-1, 1, 1, 1)
    return sqrt_acp * x0 + sqrt_m1 * noise


# ----------------------------------------------------
# 训练
# ----------------------------------------------------
def main():

    # ------------------------------------------------
    # 1. 数据
    # ------------------------------------------------
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    dataset = CIFAR10('./data', train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=BATCH, shuffle=True, num_workers=2)

    # ------------------------------------------------
    # 2. Tiny-UNet 模型
    # ------------------------------------------------
    model = get_model("diffusion").to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    mse_loss = nn.MSELoss()

    print(f"[INFO] Training Tiny-UNet Diffusion on {DEVICE}")

    # ------------------------------------------------
    # 3. 训练循环
    # ------------------------------------------------
    for epoch in range(1, EPOCHS + 1):
        total_loss = 0

        for x0, _ in loader:
            x0 = x0.to(DEVICE)

            # ---------------------------
            # 随机选择时间 t
            # ---------------------------
            t = torch.randint(0, T, (x0.size(0),), device=DEVICE).long()

            # ---------------------------
            # 采样真实噪声 ε
            # ---------------------------
            noise = torch.randn_like(x0)

            # ---------------------------
            # 得到 x_t = sqrt(...)*x0 + sqrt(...)*noise
            # ---------------------------
            x_t = q_sample(x0, t, noise)

            # ---------------------------
            # Tiny-UNet 预测噪声 εθ
            # ---------------------------
            pred_noise = model(x_t, t)

            # ---------------------------
            # MSE(εθ, ε)
            # ---------------------------
            loss = mse_loss(pred_noise, noise)
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"[Epoch {epoch}/{EPOCHS}] Loss={total_loss/len(loader):.4f}")

    # ------------------------------------------------
    # 4. 保存模型
    # ------------------------------------------------
    os.makedirs("./artifacts", exist_ok=True)
    torch.save(model.state_dict(), "./artifacts/diffusion_tinyunet.pt")

    print("[INFO] Tiny-UNet Diffusion training complete.")
    print("[INFO] Saved to ./artifacts/diffusion_tinyunet.pt")


if __name__ == "__main__":
    main()