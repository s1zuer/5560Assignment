import torch
import matplotlib.pyplot as plt
from torchvision import transforms

# 随机采样潜在向量，通过解码器生成图像并展示
def generate_samples(model, device: str = "cpu", num_samples: int = 10):
    model.to(device).eval()
    assert hasattr(model, "decode"), "VAE model must implement decode(z)"
    # 猜测 z 维度
    z_dim = next(p for n, p in model.named_parameters() if "fc_mu" in n).shape[0]
    z = torch.randn(num_samples, z_dim, device=device)
    with torch.no_grad():
        x_hat = model.decode(z).clamp(0, 1).cpu()

    inv = transforms.Normalize(mean=[0.,0.,0.], std=[1.,1.,1.])
    cols = min(5, num_samples)
    rows = (num_samples + cols - 1) // cols
    plt.figure(figsize=(2.2*cols, 2.2*rows))
    for i in range(num_samples):
        img = x_hat[i].permute(1, 2, 0).numpy()
        plt.subplot(rows, cols, i+1)
        plt.imshow(img)
        plt.axis('off')
    plt.suptitle("VAE Generated Samples")
    plt.tight_layout()
    plt.show()