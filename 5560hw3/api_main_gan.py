import io
import torch
from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import Response
from torchvision.utils import make_grid
from PIL import Image

from helper_lib.model import get_model


# -------- 设备选择 --------
def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

device = get_device()
print(f"[INFO] API using device: {device}")


# -------- 初始化 FastAPI --------
app = FastAPI(
    title="GAN Digit Generator API",
    description="Generate MNIST-like digits using our DCGAN model",
    version="1.0",
)


# -------- 加载 GAN 模型（全局只做一次）--------
gan_model = get_model("gan")
gan_model.to(device)
gan_model.eval()

try:
    # 这里的文件必须存在：artifacts/gan_generator.pt
    # 你之前 cp 的:
    # cp artifacts/gan_generator_epoch10.pt artifacts/gan_generator.pt
    state = torch.load("./artifacts/gan_generator.pt", map_location=device)
    gan_model.generator.load_state_dict(state, strict=False)
    gan_model.generator.eval()
    print("[INFO] Loaded generator weights into API.")
except Exception as e:
    print("[ERROR] Failed to load GAN generator weights:", e)
    # 注意：我们不 raise，这样 FastAPI 还能启动
    # 如果权重真的没加载成功，后面会在调用时抛 HTTPException


def sample_digits(n: int = 16, seed: int | None = None):
    """
    取 n 张生成图，返回拼好的 grid tensor，形状 [3,H,W]，像素在[0,1]
    """
    if seed is not None:
        torch.manual_seed(seed)

    z = torch.randn(n, gan_model.z_dim, device=device)

    with torch.no_grad():
        fake_imgs = gan_model.generator(z).detach().cpu()  # [n,1,28,28]

    grid = make_grid(
        fake_imgs,
        nrow=4,            # 每行4张
        normalize=True,    # 把像素放到0~1
        pad_value=0.0
    )  # [3,H,W]
    return grid


def tensor_grid_to_png_bytes(grid_tensor: torch.Tensor) -> bytes:
    """
    把 [3,H,W] 的 tensor 转成 PNG（二进制），用 Pillow 而不是 matplotlib，
    所以不会调用 GUI，也不会被 macOS 杀。
    """
    # 限制到0~1，防止出界
    grid_clamped = torch.clamp(grid_tensor, 0.0, 1.0)
    grid_img = (grid_clamped * 255).byte()        # [3,H,W] uint8
    grid_img = grid_img.permute(1, 2, 0).numpy()  # [H,W,3] HWC

    pil_img = Image.fromarray(grid_img)

    # 放大 4 倍，Swagger 里更清楚
    upscale = 4
    pil_img = pil_img.resize(
        (pil_img.width * upscale, pil_img.height * upscale),
        resample=Image.NEAREST
    )

    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    return buf.read()


@app.get(
    "/gan/generate.png",
    summary="Generate MNIST-like digits via GAN",
    description="Returns a PNG grid of generated digits produced by the DCGAN generator",
    response_description="PNG image"
)
def generate_digits_endpoint(
    n: int = Query(16, ge=1, le=64, description="number of samples (grid cells)"),
    seed: int | None = Query(None, description="random seed (optional, for reproducibility)")
):
    # 这里我们不再用那个 requires_grad 检查了
    # 直接尝试生成；如果生成器没加载好，会在下面 except 抛500

    try:
        grid = sample_digits(n=n, seed=seed)
        png_bytes = tensor_grid_to_png_bytes(grid)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate image: {e}"
        )

    return Response(content=png_bytes, media_type="image/png")