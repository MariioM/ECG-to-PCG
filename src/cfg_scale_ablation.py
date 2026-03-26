import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import signal
from scipy.stats import pearsonr
from scipy.signal import hilbert, coherence
from tqdm import tqdm
import pandas as pd

from src.config import Config
from src.models import PCGVAE, FlowTransformer, Encoder1D
from src.dataset import get_dataloaders
from src.utils import seed_everything
from src.evaluate import calculate_snr, calculate_weighted_coherence

def run_evaluation_tests(steps, scale):
    seed_everything(Config.SEED)
    cfg_models = Config.load_model_config()
    cfg_inference = Config.load_inference_config()
    _, _, test_loader = get_dataloaders(Config)
    
    device = Config.DEVICE

    vae = PCGVAE(in_channels=1, base_channels=32, z_dim=16).to(device)
    vae.load_state_dict(torch.load(Config.CHECK_DIR / cfg_models["paths"]["vae_checkpoint"], map_location=device))
    
    ecg_encoder = Encoder1D(in_channels=1, base_channels=32, z_dim=16).to(device)
    transformer = FlowTransformer(
        in_channels=16, ecg_channels=16,
        hidden_size=cfg_models["rf"]["hidden_size"], 
        depth=cfg_models["rf"]["depth"], 
        num_heads=cfg_models["rf"]["num_heads"]
    ).to(device)
    
    checkpoint = torch.load(Config.CHECK_DIR / cfg_models['paths']['flow_checkpoint'], map_location=device)
    ecg_encoder.load_state_dict(checkpoint['ecg_encoder'])
    transformer.load_state_dict(checkpoint['transformer'])
    
    stats_mean = checkpoint['norm_stats']['mean'].to(device)
    stats_std = checkpoint['norm_stats']['std'].to(device)
    
    vae.eval(); ecg_encoder.eval(); transformer.eval()

    metrics = {
        "snr": [], "cc": [], "coh_w": [],
        "env_cc": [], "env_coh": []
    }

    print(f"Strating materics with ({len(test_loader)} batches)...")
    
    with torch.no_grad():
        for batch in tqdm(test_loader):
            ecg = batch["ecg"].to(device)
            real_pcg = batch["pcg"].to(device)
            B = ecg.shape[0]

            # Inferencia con CFG (Euler)
            cond, _ = ecg_encoder(ecg)
            uncond = torch.zeros_like(cond)
            z = torch.randn(B, 16, 312).to(device)
            dt = 1.0 / steps
            
            for i in range(steps):
                t = torch.full((B*2,), i/steps, device=device)
                z_in = torch.cat([z, z], dim=0)
                c_in = torch.cat([cond, uncond], dim=0)
                v_pred = transformer(x=z_in, t=t, c_ecg=c_in)
                v_c, v_u = v_pred.chunk(2)
                v = v_u + scale * (v_c - v_u)
                z = z + v * dt

            fake_pcg = vae.decode((z * stats_std) + stats_mean, target_len=Config.SEQ_LEN)
            
            real_np = real_pcg.squeeze(1).cpu().numpy()
            fake_np = fake_pcg.squeeze(1).cpu().numpy()

            for i in range(B):
                r, f = real_np[i], fake_np[i]
                
                metrics["snr"].append(calculate_snr(r, f))
                cc, _ = pearsonr(r, f)
                metrics["cc"].append(cc)
                metrics["coh_w"].append(calculate_weighted_coherence(r, f))
                
                env_r = np.abs(hilbert(r))
                env_f = np.abs(hilbert(f))
                ecc, _ = pearsonr(env_r, env_f)
                metrics["env_cc"].append(ecc)
                
                _, cxy = coherence(env_r, env_f, fs=2000, nperseg=256)
                metrics["env_coh"].append(np.mean(cxy))
            return metrics

if __name__ == "__main__":
    Path("cfg_scale_results").mkdir(exist_ok=True)

    steps_configs = [25]
    scale_configs = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 7]
    final_results = []
    for scale in scale_configs:
        metrics = run_evaluation_tests(25, scale)
        print(f"SNR: {np.mean(metrics['snr']):.2f}, CC: {np.mean(metrics['cc']):.4f}, Coherence: {np.mean(metrics['coh_w']):.4f}")
        res = {
            "Scale": scale,
            "SNR": round(np.mean(metrics["snr"]), 2), 
            "CC": round(np.mean(metrics['cc']), 4), 
            "Coherence": round(np.mean(metrics['coh_w']), 4)
        }
        final_results.append(res)

        pd.DataFrame(final_results).to_csv("cfg_scale_results/scale_metrics_2.csv", index=False)
    
    print("\nCFG Scale test finished")
        


