import os
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import FileResponse
import torch
from PIL import Image, UnidentifiedImageError
from torchvision import transforms
from torchvision.utils import save_image, make_grid

from helper_lib.model import get_model, SimpleEBM   # ← SimpleDiffusion 已删除
from helper_lib.diffusion_generator import DiffusionGenerator
from helper_lib.ebm_generator import EBMGenerator




CLASSES = [
    'airplane','automobile','bird','cat','deer',
    'dog','frog','horse','ship','truck'
]

ALLOWED_CONTENT = {"image/png", "image/jpeg", "image/webp"}

DEVICE = (
    "cuda" if torch.cuda.is_available() else
    "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else
    "cpu"
)

app = FastAPI(title="Unified CIFAR10 + Diffusion + EBM API")


@app.get("/")
def root():
    return {
        "message": "Welcome to CIFAR10 Deep Learning API ✨",
        "endpoints": {
            "/classify": "Classify CIFAR10 images",
            "/generate?model=diffusion": "Generate with Tiny-UNet diffusion",
            "/generate?model=ebm": "Generate with Energy-Based Model",
        }
    }


@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE}



cnn_model = None

tfm = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2023, 0.1994, 0.2010)
    )
])


@app.on_event("startup")
def load_cnn_model():
    global cnn_model
    print("📌 Loading CNN model...")

    cnn_model = get_model("cnn")   # ✔️ 加载 SimpleCNN
    state = torch.load("./artifacts/cnn_best.pt", map_location=DEVICE)

    cnn_model.load_state_dict(state)
    cnn_model.to(DEVICE)
    cnn_model.eval()

    print(f"✅ Loaded SimpleCNN on {DEVICE}")



@app.post("/classify")
async def classify(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_CONTENT:
        raise HTTPException(415, f"Unsupported Content-Type: {file.content_type}")

    try:
        file.file.seek(0)
        img = Image.open(file.file).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(400, "Invalid image file")

    x = tfm(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = cnn_model(x)
        prob = torch.softmax(logits, dim=1)[0]
        conf, pred = torch.max(prob, dim=0)

    return {
        "class": CLASSES[int(pred)],
        "confidence": float(conf)
    }




@app.get("/generate")
def generate(model: str = Query(..., description="diffusion / ebm")):
    model = model.lower()


    if model == "diffusion":
        print("🌀 Generating samples using Tiny-UNet diffusion...")

        # 1. Load model
        diff_model = get_model("diffusion").to(DEVICE)
        diff_model.load_state_dict(
            torch.load("./artifacts/diffusion_tinyunet.pt", map_location=DEVICE)
        )
        diff_model.eval()

        # 2. Sampler
        generator = DiffusionGenerator(
            model=diff_model,
            device=DEVICE,
            num_steps=200
        )

        samples = generator.sample(n=16)

        # 3. Save
        grid = make_grid(samples, nrow=4, normalize=True)
        os.makedirs("./samples", exist_ok=True)
        out_path = "./samples/diffusion_api.png"
        save_image(grid, out_path)

        return FileResponse(out_path, media_type="image/png")


    elif model == "ebm":
        print("⚡ Generating samples using EBM...")

        ebm = SimpleEBM().to(DEVICE)
        ebm.load_state_dict(torch.load("./artifacts/ebm.pth", map_location=DEVICE))
        ebm.eval()

        generator = EBMGenerator(
            model=ebm,
            device=DEVICE,
            steps=50,
            lr=0.1,
            noise_std=0.01
        )

        samples = generator.sample(n=16)

        grid = make_grid(samples, nrow=4, normalize=True)
        os.makedirs("./samples", exist_ok=True)
        out_path = "./samples/ebm_api.png"
        save_image(grid, out_path)

        return FileResponse(out_path, media_type="image/png")


    else:
        raise HTTPException(400, "Invalid model. Choose: diffusion / ebm.")
