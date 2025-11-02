import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from helper_lib.model import get_model


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    device = get_device()
    print(f"[INFO] Using device: {device}")

    # ===== 修改这里：选择最新版本的 Generator 权重 =====
    # 确保你用的是新结构训练的那次（DCGAN 版）
    state_path = "./artifacts/gan_generator_epoch10.pt"  # 或者你想要的 epoch
    print(f"[INFO] Loading weights from {state_path}")

    # ===== 初始化模型并加载参数 =====
    gan_model = get_model("gan").to(device)
    state_dict = torch.load(state_path, map_location=device)
    gan_model.generator.load_state_dict(state_dict, strict=False)

    gan_model.eval()

    # ===== 生成随机噪声并生成图像 =====
    z = torch.randn(64, gan_model.z_dim, device=device)
    with torch.no_grad():
        fake_imgs = gan_model.generator(z).detach().cpu()

    # ===== 绘制图像网格 =====
    grid = make_grid(fake_imgs, nrow=8, normalize=True)
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Generated Samples (DCGAN)")
    plt.imshow(grid.permute(1, 2, 0).numpy())
    plt.show()


if __name__ == "__main__":
    main()