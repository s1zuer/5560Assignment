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

    # KL: D_KL(q(z|x) || N(0, I))
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

import torch

def train_gan(model,
              data_loader,
              criterion,
              optimizer,
              device: str = "cpu",
              epochs: int = 10):
    """
    稳定版训练循环，目的是减轻 mode collapse：
    1. 判别器别太强 (lr 更低，在外面传进来)
    2. label smoothing：真实 = 0.9 而不是 1.0
    3. 生成器每个 batch 训练两次，鼓励多样性
    4. 每 5 个 epoch 存一个 checkpoint (方便挑最好看的版本)

    参数:
      model: 你的 GAN()，包含 generator / discriminator / z_dim
      data_loader: MNIST dataloader
      criterion: nn.BCELoss()
      optimizer: {
          "gen":  Adam(...),
          "disc": Adam(...)
      }
    """

    model.to(device)
    G = model.generator
    D = model.discriminator
    z_dim = model.z_dim

    for ep in range(1, epochs + 1):
        G.train()
        D.train()

        d_loss_running = 0.0
        g_loss_running = 0.0
        batch_count = 0

        for real_img, _ in data_loader:
            real_img = real_img.to(device)
            bs = real_img.size(0)

            # --------- label smoothing ---------
            real_label = torch.ones(bs, 1, device=device) * 0.9  # not 1.0
            fake_label = torch.zeros(bs, 1, device=device)       # still 0.0

            # =========================
            # 1. Update Discriminator
            # =========================
            optimizer["disc"].zero_grad(set_to_none=True)

            # D(real) -> want ~0.9
            out_real = D(real_img)
            d_loss_real = criterion(out_real, real_label)

            # generate fake batch (no grad to G here)
            z = torch.randn(bs, z_dim, device=device)
            fake_img = G(z).detach()
            out_fake = D(fake_img)
            d_loss_fake = criterion(out_fake, fake_label)

            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            optimizer["disc"].step()

            # =========================
            # 2. Update Generator (twice)
            # =========================
            # G tries to fool D: D(G(z)) -> 0.9
            g_loss_total_this_batch = 0.0
            for _ in range(2):  # 关键点：给G两次学习机会
                optimizer["gen"].zero_grad(set_to_none=True)

                z2 = torch.randn(bs, z_dim, device=device)
                gen_img = G(z2)
                out_gen = D(gen_img)

                target_for_g = torch.ones(bs, 1, device=device) * 0.9
                g_loss = criterion(out_gen, target_for_g)

                g_loss.backward()
                optimizer["gen"].step()

                g_loss_total_this_batch += g_loss.item()

            # 记录loss（把两次G更新的loss平均一下，方便打印）
            d_loss_running += d_loss.item()
            g_loss_running += (g_loss_total_this_batch / 2.0)
            batch_count += 1

        print(
            f"[Epoch {ep}/{epochs}] "
            f"D_loss={d_loss_running/batch_count:.4f} "
            f"G_loss={g_loss_running/batch_count:.4f}"
        )

        # 每5个epoch存checkpoint，这样你可以挑没崩掉的版本
        if ep % 5 == 0:
            ckpt_path = f"./artifacts/gan_generator_epoch{ep}.pt"
            torch.save(G.state_dict(), ckpt_path)
            print(f"[checkpoint] saved generator at {ckpt_path}")

    return model