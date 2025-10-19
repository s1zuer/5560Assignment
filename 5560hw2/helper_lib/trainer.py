import torch
import torch.nn as nn
import torch.optim as optim

# ---- 通用分类训练循环（Module 4 活动） ----
def train_model(model, data_loader, criterion, optimizer,
                device: str = "cpu", epochs: int = 10):
    model.to(device)
    for ep in range(1, epochs+1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
        print(f"[Epoch {ep}/{epochs}] loss={running_loss/total:.4f} acc={correct/total:.4f}")
    return model

# ---- VAE 训练（Module 5 活动） ----
def vae_loss_fn(x_hat, x, mu, logvar, recon="bce", beta=1.0):
    if recon == "bce":
        recon_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction="sum")
    else:
        recon_loss = nn.functional.mse_loss(x_hat, x, reduction="sum")
    # KL: D_KL(q(z|x) || N(0, I)) = -0.5 * Σ(1 + logσ² - μ² - σ²)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl, recon_loss, kl

def train_vae_model(model, data_loader, optimizer,
                    device: str = "cpu", epochs: int = 10,
                    recon: str = "bce", beta: float = 1.0):
    model.to(device)
    for ep in range(1, epochs+1):
        model.train()
        total, rec_sum, kl_sum = 0.0, 0.0, 0.0
        for x, _ in data_loader:
            x = x.to(device)
            x_hat, mu, logvar = model(x)
            loss, rec, kl = vae_loss_fn(x_hat, x, mu, logvar, recon=recon, beta=beta)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += loss.item()
            rec_sum += rec.item()
            kl_sum += kl.item()
        n = len(data_loader.dataset)
        print(f"[Epoch {ep}/{epochs}] total={(total/n):.4f} recon={(rec_sum/n):.4f} kl={(kl_sum/n):.4f}")
    return model