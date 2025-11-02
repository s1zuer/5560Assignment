import torch
import matplotlib.pyplot as plt
from torchvision import transforms

# ===== VAE 采样可视化 =====
def generate_vae_samples(model, device: str = "cpu", num_samples: int = 10):
    """
    输入: 训练好的 VAE 模型
    输出: 采样生成的图片网格 (32x32彩色)
    """
    model.to(device).eval()
    assert hasattr(model, "decode"), "VAE model must implement decode(z)"

    # 猜测潜在维度 z_dim：通过 fc_mu 找到维度
    # (你之前的逻辑基本就是这样)
    z_dim = next(p for n, p in model.named_parameters() if "fc_mu" in n).shape[0]

    z = torch.randn(num_samples, z_dim, device=device)
    with torch.no_grad():
        x_hat = model.decode(z).clamp(0, 1).cpu()  # [0,1]

    cols = min(5, num_samples)
    rows = (num_samples + cols - 1) // cols
    plt.figure(figsize=(2.2*cols, 2.2*rows))
    for i in range(num_samples):
        img = x_hat[i].permute(1, 2, 0).numpy()  # C,H,W -> H,W,C
        plt.subplot(rows, cols, i+1)
        plt.imshow(img)
        plt.axis('off')
    plt.suptitle("VAE Generated Samples")
    plt.tight_layout()
    plt.show()

# ===== GAN 采样可视化 =====
def generate_gan_samples(model, device: str = "cpu", num_samples: int = 16):
    """
    输入:
        model: 可以是
            - 我们定义的 GAN() 实例 (包含 model.generator, model.z_dim)
            - 或者直接传 generator 子网 (只要它有 forward(z) 和 z_dim)
    输出:
        显示一批 MNIST 风格的手写数字 (灰度 28x28)，用 plt 网格画出来
    """

    # 判断你传进来的是 GAN 整体还是单独的 generator
    if hasattr(model, "generator"):
        G = model.generator
        z_dim = model.z_dim
    else:
        G = model
        # 如果 generator 本身就保存了 z_dim，我们读它；否则默认100
        z_dim = getattr(model, "z_dim", 100)

    G.to(device).eval()

    with torch.no_grad():
        z = torch.randn(num_samples, z_dim, device=device)
        fake_imgs = G(z).cpu()  # shape: [N, 1, 28, 28], 值范围大概在[-1,1] (tanh)
        fake_imgs = (fake_imgs + 1) / 2.0  # 把 [-1,1] 映射到 [0,1] 方便imshow

    cols = min(4, num_samples)
    rows = (num_samples + cols - 1) // cols
    plt.figure(figsize=(2.0*cols, 2.0*rows))
    for i in range(num_samples):
        img = fake_imgs[i, 0].numpy()  # 取出单通道
        plt.subplot(rows, cols, i+1)
        plt.imshow(img, cmap="gray")
        plt.axis('off')
    plt.suptitle("GAN Generated Samples")
    plt.tight_layout()
    plt.show()