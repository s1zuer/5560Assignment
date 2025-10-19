import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms as T
from helper_lib.model import get_model
from helper_lib.data_loader import get_data_loader

CLASSES = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

# CIFAR-10 的均值和方差（和 data_loader 保持一致）
MEAN = (0.4914, 0.4822, 0.4465)
STD  = (0.2470, 0.2435, 0.2616)
INV_NORM = T.Normalize(mean=[-m/s for m, s in zip(MEAN, STD)],
                       std=[1/s for s in STD])

def imshow_denorm(x):
    # x: CxHxW，已经 Normalize 过
    x = INV_NORM(x).clamp(0, 1).cpu()         # 反标准化并裁到[0,1]
    img = x.permute(1, 2, 0).numpy()
    plt.imshow(img)
    plt.axis("off")

def visualize_predictions(model_path="./artifacts/cnn_best.pt",
                          num_images=8,
                          device="cpu"):
    model = get_model("CNN")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()

    test_loader = get_data_loader("../data", batch_size=num_images, train=False)
    images, labels = next(iter(test_loader))
    images = images.to(device)

    with torch.no_grad():
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        confs, preds = torch.max(probs, 1)

    plt.figure(figsize=(10, 8))
    for i in range(num_images):
        plt.subplot(2, 4, i + 1)
        imshow_denorm(images[i])
        plt.title(f"True: {CLASSES[labels[i]]}\nPred: {CLASSES[preds[i]]} ({confs[i].item():.2f})")
    plt.tight_layout()
    plt.savefig("./artifacts/results_visualization.png", dpi=160)  # 方便提交
    plt.show()