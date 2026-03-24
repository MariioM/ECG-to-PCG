import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path

from src.config import Config
from src.models import PCGVAE
from src.dataset import get_dataloaders
from src.utils import seed_everything

# VAE Loss Function
def vae_loss_fn(recon_x, x, mu, logvar, beta=0.0001):
    recon_loss = F.l1_loss(recon_x, x, reduction="mean")

    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kl_loss = kl_loss / x.numel()

    total_loss = recon_loss + beta * kl_loss
    return total_loss, recon_loss, kl_loss

def train_vae():
    # Load the vae config from yaml file
    cfg = Config.load_model_config()
    seed_everything(Config.SEED)

    # Load the dataloaders
    train_loader, val_loader, _ = get_dataloaders(Config)

    # Instance the VAE and optimizer
    vae = PCGVAE(
        in_channels=cfg["vae"]["in_channels"],
        base_channels=cfg["vae"]["base_channels"],
        z_dim=cfg["vae"]["z_dim"]
    ).to(Config.DEVICE)

    optimizer = optim.AdamW(
        vae.parameters(),
        lr=float(cfg["vae_training"]["lr"]),
        weight_decay=float(cfg["vae_training"]["weight_decay"])
    )

    # Setup the checkpoint directory
    Config.CHECK_DIR.mkdir(parents=True, exist_ok=True)
    best_model_path = Config.CHECK_DIR / cfg["paths"]["vae_checkpoint"]

    # Setup the loss history for plots and checkpoints
    best_val_loss = float("inf")
    history = {
        "train_loss": [],
        "val_loss": []
    }

    # Setup the training hyperparameters
    VAE_EPOCHS = cfg["vae_training"]["epochs"]
    KL_WARMUP_EPOCHS = cfg["vae_training"]["kl_warmup_epochs"]
    MAX_BETA = cfg["vae_training"]["max_beta"]

    print("Begining VAE training")
    print(f"Device: {Config.DEVICE} | Model: PCGVAE | Epochs: {VAE_EPOCHS} | LR: {cfg["vae_training"]["lr"]} | KL Warmup: {KL_WARMUP_EPOCHS} epochs")

    # Setup the KL warmup
    for epoch in range(1, VAE_EPOCHS + 1):
        if epoch < KL_WARMUP_EPOCHS:
            current_beta = MAX_BETA * (epoch / KL_WARMUP_EPOCHS)
        else:
            current_beta = MAX_BETA
    
        vae.train()
        train_loss_acum = 0
        recon_loss_acum = 0
        kl_loss_acum = 0

        pbar = tqdm(train_loader, desc=f"Ep {epoch}/{VAE_EPOCHS} [Train]")

        # Train VAE
        for batch in pbar:
            x = batch["pcg"].to(Config.DEVICE)

            optimizer.zero_grad()

            recon_x, mu, logvar = vae(x)
            loss, l_rec, l_kl = vae_loss_fn(recon_x, x, mu, logvar, beta = current_beta)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=float(cfg["vae_training"]["clip_grad"]))
            optimizer.step()

            train_loss_acum += loss.item()
            recon_loss_acum += l_rec.item()
            kl_loss_acum += l_kl.item()

            pbar.set_postfix({"L_Rec": f"{l_rec.item():.4f}", "Beta": f"{current_beta:.5f}"})

        # Train metrics
        avg_train_loss = train_loss_acum / len(train_loader)
        avg_recon = recon_loss_acum / len(train_loader)
        avg_kl = kl_loss_acum / len(train_loader)

        # VAE Eval
        vae.eval()
        val_loss_acum = 0

        with torch.no_grad():
            for batch in val_loader:
                x = batch["pcg"].to(Config.DEVICE)
                recon_x, mu, log_var = vae(x)
                loss, _, _ = vae_loss_fn(recon_x, x, mu, logvar, beta=current_beta)
                val_loss_acum += loss.item()
        # Validation Metric
        avg_val_loss = val_loss_acum / len(val_loader)

        # Add validation and train losses to the history
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)

        print(f"Ep {epoch} | Train Loss: {avg_train_loss:.5f} (Rec: {avg_recon:.5f} | KL: {avg_kl:.5f}) | Val Loss: {avg_val_loss:.5f}")

        # Save the best checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(vae.state_dict(), best_model_path)
            print(f"New best model saved (Val loss: {best_val_loss:.5f})")

    print("VAE Trainined")

if __name__ == "__main__":
    train_vae()





