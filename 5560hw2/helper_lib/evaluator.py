import torch
import torch.nn as nn

@torch.no_grad()
def evaluate_model(model, data_loader, criterion, device: str = "cpu"):
    model.to(device).eval()
    loss_sum, correct, total = 0.0, 0, 0
    for x, y in data_loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        loss_sum += loss.item() * x.size(0)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return loss_sum/total, correct/total