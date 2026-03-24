import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.config import Config
from src.models import PCGVAE, Encoder1D, FlowTransformer
from src.dataset import get_dataloaders

def save_clean_plot(data, filename, color='black'):
    """Guarda un plot sin ejes, sin márgenes y ocupando todo el rectángulo."""
    fig, ax = plt.subplots(figsize=(4, 2))

    ax.plot(data, color=color, linewidth=1.5)

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor('black')
        spine.set_linewidth(2.0)
    
    ax.set_xlim(0, len(data) - 1)
    
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    
    output_path = Path("diagram_assets")
    output_path.mkdir(exist_ok=True)
    plt.savefig(output_path / filename, bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.close()
    print(f"Asset guardado: {filename}")

def generate_assets():
    device = Config.DEVICE
    cfg_models = Config.load_model_config()
    cfg_inference = Config.load_inference_config()
    _, _, test_loader = get_dataloaders(Config)
    
    vae = PCGVAE(in_channels=1, base_channels=32, z_dim=16).to(device)
    vae.load_state_dict(torch.load(Config.CHECK_DIR / cfg_models["paths"]["vae_checkpoint"], map_location=device))
    ecg_encoder = Encoder1D(in_channels=1, base_channels=32, z_dim=16).to(device)
    
    transformer = FlowTransformer(
        in_channels=32, 
        hidden_size=cfg_models["rf"]["hidden_size"], 
        depth=cfg_models["rf"]["depth"], 
        num_heads=cfg_models["rf"]["num_heads"]
    ).to(device)
    
    flow_ckpt = torch.load(Config.CHECK_DIR / cfg_models['paths']['flow_checkpoint'], map_location=device)
    transformer.load_state_dict(flow_ckpt['transformer'])
    ecg_encoder.load_state_dict(flow_ckpt['ecg_encoder'])
    
    stats_mean = flow_ckpt['norm_stats']['mean'].to(device)
    stats_std = flow_ckpt['norm_stats']['std'].to(device)

    batch = next(iter(test_loader))
    ecg_raw = batch["ecg"][0:1].to(device) 
    pcg_raw = batch["pcg"][0:1].to(device)

    vae.eval()
    ecg_encoder.eval()
    transformer.eval()

    with torch.no_grad():
        # Raw train signals
        save_clean_plot(ecg_raw.cpu().numpy().flatten(), "1_ecg_raw.png", color='#2c3e50')
        save_clean_plot(pcg_raw.cpu().numpy().flatten(), "1_pcg_raw.png", color='#16a085')

        # Latent spaces
        # ECG feature space
        cond_ecg, _ = ecg_encoder(ecg_raw)
        save_clean_plot(cond_ecg.cpu().numpy().flatten()[:512], "2_ecg_features.png", color='#2980b9')
        
        # PCG Latent Space
        z1 = vae.encode(pcg_raw)
        save_clean_plot(z1.cpu().numpy().flatten()[:512], "2_pcg_z1_latent.png", color='#8e44ad')


        # z0
        z0 = torch.randn_like(z1)
        save_clean_plot(z0.cpu().numpy().flatten()[:512], "3_z0_noise.png", color='#7f8c8d')
        
        # zt
        t_interp = 0.5
        zt = t_interp * z1 + (1 - t_interp) * z0
        save_clean_plot(zt.cpu().numpy().flatten()[:512], "3_zt_interp.png", color='#e67e22')

        # Generated PCG
        steps = cfg_inference["rf_solver"]["steps"]
        z = z0
        dt = 1.0 / steps
        for i in range(steps):
            t_val = i / steps
            t_env = torch.full((1,), t_val, device=device)
            v_pred = transformer(torch.cat([z, cond_ecg], dim=1), t_env)
            z = z + v_pred * dt
        
        pcg_fake = vae.decode((z * stats_std) + stats_mean, target_len=Config.SEQ_LEN)
        save_clean_plot(pcg_fake.cpu().numpy().flatten(), "4_pcg_generated.png", color='#d62728')

if __name__ == "__main__":
    generate_assets()