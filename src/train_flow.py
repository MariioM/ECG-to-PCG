import torch
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path

from src.config import Config
from src.models import PCGVAE, FlowTransformer, Encoder1D
from src.dataset import get_dataloaders
from src.utils import seed_everything

def calculate_latent_stats(vae, loader, device, num_batches=20):
    vae.eval()
    all_latents = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            pcg = batch["pcg"].to(device)
            z = vae.encode(pcg) 
            all_latents.append(z.cpu())
            if i >= num_batches: break
    
    all_latents = torch.cat(all_latents, dim=0)
    return all_latents.mean(), all_latents.std()

def train_flow():
    seed_everything(Config.SEED)
    cfg_model = Config.load_model_config()
    train_loader, val_loader, _ = get_dataloaders(Config)

    vae = PCGVAE(
        in_channels=cfg_model["vae"]["in_channels"],
        base_channels=cfg_model["vae"]["base_channels"],
        z_dim=cfg_model["vae"]["z_dim"]
    ).to(Config.DEVICE)
    
    vae_path = Config.CHECK_DIR / cfg_model["paths"]["vae_checkpoint"]
    vae.load_state_dict(torch.load(vae_path, map_location=Config.DEVICE))
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False

    STATS_MEAN, STATS_STD = calculate_latent_stats(vae, train_loader, Config.DEVICE)
    print(f"Mean: {STATS_MEAN:.4f} | Std: {STATS_STD:.4f}")

    ecg_encoder = Encoder1D(in_channels=1, base_channels=32, z_dim=16).to(Config.DEVICE)
    transformer = FlowTransformer(in_channels=32, hidden_size=256, depth=8, num_heads=4).to(Config.DEVICE)

    optimizer = optim.AdamW(
        list(ecg_encoder.parameters()) + list(transformer.parameters()), 
        lr=5e-5, weight_decay=1e-4
    )

    best_flow_path = Config.CHECK_DIR / cfg_model["paths"]["flow_checkpoint"]
    best_val_loss = float("inf")
    CFG_PROB = cfg_model["rf"]["cfg_prob"]
    EPOCHS = cfg_model["rf_training"]["epochs"]

    for epoch in range(1, EPOCHS + 1):
        ecg_encoder.train()
        transformer.train()
        train_loss_acc = 0
        
        pbar = tqdm(train_loader, desc=f"Flow Ep {epoch}/{EPOCHS}")
        for batch in pbar:
            ecg = batch["ecg"].to(Config.DEVICE)
            pcg = batch["pcg"].to(Config.DEVICE)
            
            if pcg.abs().max() > 1.5: pcg = pcg / (pcg.abs().max() + 1e-6)
            if ecg.abs().max() > 1.5: ecg = ecg / (ecg.abs().max() + 1e-6)

            optimizer.zero_grad()
            
            with torch.no_grad():
                z1 = (vae.encode(pcg) - STATS_MEAN.to(Config.DEVICE)) / STATS_STD.to(Config.DEVICE)
            
            cond_ecg, _ = ecg_encoder(ecg)
            
            if torch.rand(1).item() < CFG_PROB:
                cond_ecg = torch.zeros_like(cond_ecg)
                
            B = z1.shape[0]
            t = torch.rand(B, device=Config.DEVICE)
            z0 = torch.randn_like(z1)
            
            t_exp = t.view(B, 1, 1)
            z_t = t_exp * z1 + (1 - t_exp) * z0
            target_v = z1 - z0
            
            # CAMBIADOSDASDFWER23ER
            pred_v = transformer(model_in, t)
            
            loss = F.mse_loss(pred_v, target_v)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(ecg_encoder.parameters()) + list(transformer.parameters()), 1.0)
            optimizer.step()
            
            train_loss_acc += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.5f}"})

        avg_val_loss = train_loss_acc / len(train_loader)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'ecg_encoder': ecg_encoder.state_dict(),
                'transformer': transformer.state_dict(),
                'norm_stats': {'mean': STATS_MEAN, 'std': STATS_STD}
            }, best_flow_path)

if __name__ == "__main__":
    train_flow()