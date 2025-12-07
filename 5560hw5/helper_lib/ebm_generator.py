import torch
import torch.nn as nn
import torch.nn.functional as F


class EBMGenerator:
    """
    EBM 采样器：推理阶段使用的 Langevin dynamics
    必须和训练时使用的参数保持一致，才能避免雪花屏。
    """

    def __init__(self, model, device="cpu",
                 steps=60,            # ← 与训练一致
                 lr=0.01,            # ← 训练使用 0.01，而不是 0.1
                 noise_std=0.01):    # 与训练一致
        self.model = model.to(device)
        self.device = device
        self.steps = steps
        self.lr = lr
        self.noise_std = noise_std

    @torch.no_grad()
    def init_noise(self, n=16):
        """初始化随机图像"""
        return torch.randn(n, 3, 32, 32, device=self.device)

    def sample(self, n=16):
        """
        Langevin 动力学生成（稳定版）
        """
        x = self.init_noise(n).requires_grad_(True)

        for _ in range(self.steps):
            energy = self.model(x).sum()
            grad = torch.autograd.grad(energy, x)[0]

            # 下降能量（与训练相同）
            x = x - self.lr * grad

            # Langevin 噪声
            x = x + self.noise_std * torch.randn_like(x)

            # ★ 必须 clamp，否则会发散成雪花
            x = x.clamp(-2.0, 2.0)

            x = x.detach().requires_grad_(True)

        # 输出 [-1,1]，API 会自动 normalize
        return x