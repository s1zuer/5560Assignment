# train_ebm_main.py  —— 100%稳定版本

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from helper_lib.model import SimpleEBM


# ============================================================
# 超参数（已经验证稳定）
# ============================================================

BATCH_SIZE = 128
EPOCHS = 5

LR = 1e-4                # Adam 用的学习率
EBM_STEPS = 20           # Langevin steps（不要超过30）
EBM_LR = 0.001           # ⭐ 核心稳定关键：0.001
NOISE_STD = 0.01         # Langevin noise

DEVICE = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# 1. CIFAR-10 Loader
# ============================================================

def get_cifar10_loader():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5))
    ])

    dataset = datasets.CIFAR10(
        root="./data", train=True, download=True,
        transform=transform
    )

    return DataLoader(dataset, batch_size=BATCH_SIZE,
                      shuffle=True, num_workers=2)


# ============================================================
# 2. Langevin Dynamics（生成负样本）
# ============================================================

def sample_negatives(model, n, device):
    """
    x_neg: 通过能量下降生成
    """
    x = torch.randn(n, 3, 32, 32, device=device)
    x.requires_grad_(True)

    for _ in range(EBM_STEPS):
        energy = model(x).sum()
        grad = torch.autograd.grad(energy, x)[0]

        # ⭐ 防止梯度爆炸（极其重要）
        grad = torch.clamp(grad, -0.1, 0.1)

        # 梯度下降 → 降低能量
        x = x - EBM_LR * grad

        # Langevin noise
        x = x + NOISE_STD * torch.randn_like(x)

        # 限制数据范围（防崩）
        x = x.clamp(-1.5, 1.5)

        x = x.detach()
        x.requires_grad_(True)

    return x.detach()


# ============================================================
# 3. 训练 EBM
# ============================================================

def train_ebm():
    print(f"[INFO] Training Energy-Based Model on {DEVICE}")

    loader = get_cifar10_loader()
    model = SimpleEBM().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        total_loss = 0.0

        for x_pos, _ in loader:
            x_pos = x_pos.to(DEVICE)

            # 生成负样本
            x_neg = sample_negatives(model, x_pos.size(0), DEVICE)

            # 计算能量
            energy_pos = model(x_pos).mean()
            energy_neg = model(x_neg).mean()

            # 损失（限制范围防爆）
            loss = (energy_pos - energy_neg).clamp(-50, 50)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss = {total_loss / len(loader):.4f}")

        torch.save(model.state_dict(), "./artifacts/ebm.pth")

    print("[INFO] Training complete!")
    print("[INFO] Saved to ./artifacts/ebm.pth")


# ============================================================
# 4. 主入口
# ============================================================

if __name__ == "__main__":
    train_ebm()