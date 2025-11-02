import os
import torch

def save_model(model, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)

def load_model(model, path: str, device: str = "cpu"):
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    return model