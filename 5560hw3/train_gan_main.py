import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from helper_lib.model import get_model
from helper_lib.trainer import train_gan

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def main():
    device = get_device()
    print(f"[INFO] Using device: {device}")

    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_set = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=tfm  # <- 注意别拼错，保持是 tfm
    )

    train_loader = DataLoader(
        train_set,
        batch_size=64,      # 64 对 mps 来说更稳
        shuffle=True,
        drop_last=True
    )

    gan_model = get_model("gan")
    print("[INFO] Created GAN model")

    criterion = nn.BCELoss()

    opt = {
        "gen":  optim.Adam(
            gan_model.generator.parameters(),
            lr=2e-4,
            betas=(0.5, 0.999)
        ),
        "disc": optim.Adam(
            gan_model.discriminator.parameters(),
            lr=1e-4,
            betas=(0.5, 0.999)
        )
    }

    gan_model = train_gan(
        model=gan_model,
        data_loader=train_loader,
        criterion=criterion,
        optimizer=opt,
        device=device,
        epochs=15   # 别用30了，15够出好图而且少一半时间
    )

    save_path = "./artifacts/gan_generator.pt"
    torch.save(gan_model.generator.state_dict(), save_path)
    print(f"[✅ DONE] Saved generator weights to {save_path}")

if __name__ == "__main__":
    main()