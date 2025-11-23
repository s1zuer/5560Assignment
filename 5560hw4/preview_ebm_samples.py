import torch
from torchvision.utils import save_image, make_grid

from helper_lib.model import SimpleEBM
from helper_lib.ebm_generator import EBMGenerator


DEVICE = "mps" if torch.backends.mps.is_available() else \
         "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH = "./artifacts/ebm.pth"


def preview_ebm_samples():
    print(f"[INFO] Generating EBM samples on {DEVICE}")


    model = SimpleEBM().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    generator = EBMGenerator(
        model=model,
        device=DEVICE,
        steps=50,       # Langevin steps
        lr=0.1,
        noise_std=0.01
    )


    samples = generator.sample(n=16)


    grid = make_grid(samples, nrow=4, normalize=True)


    save_image(grid, "./samples/ebm_sample.png")

    print("[INFO] Saved to ./samples/ebm_sample.png")


if __name__ == "__main__":
    preview_ebm_samples()
