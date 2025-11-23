import os
import torch
from torchvision.utils import save_image, make_grid

from helper_lib.model import get_model
from helper_lib.diffusion_generator import DiffusionGenerator

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

MODEL_PATH = "./artifacts/diffusion_tinyunet.pt"


def preview_diffusion_samples():
    print(f"[INFO] Generating Diffusion samples on {DEVICE}")

    # 1. Tiny-UNet 模型
    model = get_model("diffusion").to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # 2. 创建采样器（200 steps → 与训练一致）
    generator = DiffusionGenerator(
        model=model,
        device=DEVICE,
        num_steps=200,
    )

    # 3. 生成 4x4 = 16 张图像
    samples = generator.sample(n=16)

    # 4. 拼成网格保存
    os.makedirs("./samples", exist_ok=True)
    grid = make_grid(samples, nrow=4, normalize=True)
    out_path = "./samples/diffusion_tinyunet_sample.png"
    save_image(grid, out_path)

    print(f"[INFO] Saved to {out_path}")


if __name__ == "__main__":
    preview_diffusion_samples()