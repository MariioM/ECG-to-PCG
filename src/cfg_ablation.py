import torch
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
from scipy.stats import pearsonr

from src.config import Config
from src.models import PCGVAE, FlowTransformer, Encoder1D
from src.dataset import get_dataloaders
from src.utils import seed_everything
from src.evaluate import calculate_weighted_coherence

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

def train_and_evaluate_cfg(cfg_prob, epochs=300):
    seed_everything(Config.SEED)
    cfg_model_conf = Config.load_model_config()
    train_loader, val_loader, test_loader = get_dataloaders(Config)
    device = Config.DEVICE

    vae = PCGVAE(in_channels=1, base_channels=32, z_dim=16).to(device)
    vae.load_state_dict(torch.load(Config.CHECK_DIR / cfg_model_conf["paths"]["vae_checkpoint"], map_location=device))
    vae.eval()

    MEAN, STD = calculate_latent_stats(vae, train_loader, device)
    
    ecg_encoder = Encoder1D(in_channels=1, base_channels=32, z_dim=16).to(device)
    transformer = FlowTransformer(in_channels=32, hidden_size=256, depth=8, num_heads=4).to(device)
    optimizer = optim.AdamW(list(ecg_encoder.parameters()) + list(transformer.parameters()), lr=5e-5)

    print(f"\n>>> Iniciando entrenamiento: CFG_PROB = {cfg_prob}")
    
    for epoch in tqdm(range(1, epochs + 1), desc=f"Config CFG {cfg_prob}"):
        ecg_encoder.train(); transformer.train()
        train_loss = 0
        
        for batch in train_loader:
            ecg = batch["ecg"].to(device)
            pcg = batch["pcg"].to(device)
            
            optimizer.zero_grad()
            with torch.no_grad():
                z1 = (vae.encode(pcg) - MEAN.to(device)) / STD.to(device)
            
            cond_ecg, _ = ecg_encoder(ecg)
            
            if torch.rand(1).item() < cfg_prob:
                cond_ecg = torch.zeros_like(cond_ecg)
                
            z0 = torch.randn_like(z1)
            t = torch.rand(z1.shape[0], device=device)
            z_t = t.view(-1, 1, 1) * z1 + (1 - t.view(-1, 1, 1)) * z0
            
            pred_v = transformer(torch.cat([z_t, cond_ecg], dim=1), t)
            loss = F.mse_loss(pred_v, z1 - z0)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

    ecg_encoder.eval(); transformer.eval()
    all_cc, all_coh = [], []
    
    with torch.no_grad():

        batch = next(iter(test_loader))
        ecg = batch["ecg"].to(device)
        real_pcg = batch["pcg"].to(device)

        z = torch.randn(ecg.shape[0], 16, 312).to(device)
        cond, _ = ecg_encoder(ecg)
        uncond = torch.zeros_like(cond)
        steps = 50
        for i in range(steps):
            t_val = i/steps
            t_env = torch.full((ecg.shape[0]*2,), t_val, device=device)
            v_pred = transformer(torch.cat([torch.cat([z,z]), torch.cat([cond,uncond])], dim=1), t_env)
            v_c, v_u = v_pred.chunk(2)
            v = v_u + 2.5 * (v_c - v_u) 
            z = z + v * (1.0/steps)
            
        fake_pcg = vae.decode((z * STD.to(device)) + MEAN.to(device), target_len=10000)

        plt.figure(figsize=(10, 4))
        plt.plot(fake_pcg[0,0].cpu().numpy(), color='red', label=f'CFG Drop {cfg_prob}')
        plt.title(f"Generationg with CFG {cfg_prob}")
        plt.legend()
        plt.savefig(f"cfg_study_results/plot_cfg_{cfg_prob}.png")
        plt.close()

        for i in range(ecg.shape[0]):
            r = real_pcg[i,0].cpu().numpy()
            f = fake_pcg[i,0].cpu().numpy()
            cc, _ = pearsonr(r, f)
            all_cc.append(cc)
            all_coh.append(calculate_weighted_coherence(r, f))

    return {"cfg_prob": cfg_prob, "mean_cc": np.mean(all_cc), "mean_coh": np.mean(all_coh)}

if __name__ == "__main__":
    Path("cfg_study_results").mkdir(exist_ok=True)

    configs_to_test = [0.0, 0.1, 0.15, 0.25, 0.35, 0.5, 0.7]
    final_results = []

    for p in configs_to_test:
        res = train_and_evaluate_cfg(cfg_prob=p, epochs=300)
        final_results.append(res)
        
        pd.DataFrame(final_results).to_csv("cfg_study_results/metrics.csv", index=False)

    print("\nCFG Ablation Compelted'cfg_study_results'")