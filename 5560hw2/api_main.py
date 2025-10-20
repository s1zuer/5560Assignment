from fastapi import FastAPI, UploadFile, File, HTTPException
import torch
from PIL import Image, UnidentifiedImageError
from torchvision import transforms
from helper_lib.model import get_model

CLASSES = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
ALLOWED_CONTENT = {"image/png", "image/jpeg", "image/webp"}

app = FastAPI(title="CIFAR10 CNN API")

@app.get("/")
def root():
    return {"message": "Welcome to CIFAR10 CNN API! Visit /docs for usage."}

@app.get("/health")
def health():
    return {"status": "ok"}

# 设备选择（可选 mps）
device = ("cuda" if torch.cuda.is_available() else
          "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else
          "cpu")

model = None
tfm = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4914,0.4822,0.4465), std=(0.2023,0.1994,0.2010)),
])

@app.on_event("startup")
def load_model_once():
    global model
    model = get_model("CNN")
    state = torch.load("./artifacts/cnn_best.pt", map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    print(f"✅ Loaded model on {device}")

@app.post("/classify")
async def classify(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_CONTENT:
        raise HTTPException(415, f"Unsupported Content-Type: {file.content_type}")

    try:
        file.file.seek(0)
        img = Image.open(file.file).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(400, "Invalid image file")

    x = tfm(img).unsqueeze(0)
    with torch.no_grad():
        logits = model(x.to(device))
        prob = torch.softmax(logits, dim=1)[0]
        conf, pred = torch.max(prob, dim=0)

    return {"class": CLASSES[int(pred)], "confidence": float(conf)}