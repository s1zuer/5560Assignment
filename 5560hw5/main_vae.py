import torch
import torch.optim as optim
from helper_lib.data_loader import get_data_loader
from helper_lib.model import get_model
from helper_lib.trainer import train_vae_model
from helper_lib.generator import generate_samples

device = "cuda" if torch.cuda.is_available() else "cpu"

train_loader = get_data_loader("./data", batch_size=128, train=True)
vae = get_model("VAE")

optimizer = optim.Adam(vae.parameters(), lr=1e-3)
train_vae_model(vae, train_loader, optimizer, device=device, epochs=10, recon="bce", beta=1.0)

generate_samples(vae, device=device, num_samples=10)