import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import signal
from scipy.stats import pearsonr
from scipy.signal import hilbert, coherence

from src.config import Config
from src.models import PCGVAE, FlowTransformer, Encoder1D
from src.dataset import get_dataloaders
from src.utils import seed_everything

def calculate_snr(original, reconstructed):
    noise = original - reconstructed
    p_signal = np.sum(original ** 2)
    p_noise = np.sum(noise ** 2)
    if p_noise < 1e-10: return 100.0 
    return 10 * np.log10(p_signal / (p_noise + 1e-10))

def calculate_weighted_coherence(original, reconstructed, fs=2000):
    f, Cxy = signal.coherence(original, reconstructed, fs=fs, nperseg=256)
    f_psd, Pxx = signal.welch(original, fs=fs, nperseg=256)
    weighted_coh = np.sum(Pxx * Cxy) / (np.sum(Pxx) + 1e-10)
    return weighted_coh


def run_evaluation():
    seed_everything(Config.SEED)
    cfg_models = Config.load_model_config()
    cfg_inference = Config.load_inference_config()
    _, _, test_loader = get_dataloaders(Config)
    
    device = Config.DEVICE

    vae = PCGVAE(in_channels=1, base_channels=32, z_dim=16).to(device)
    vae.load_state_dict(torch.load(Config.CHECK_DIR / cfg_models["paths"]["vae_checkpoint"], map_location=device))
    
    ecg_encoder = Encoder1D(in_channels=1, base_channels=32, z_dim=16).to(device)
    transformer = FlowTransformer(
        in_channels=32, 
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

    steps = cfg_inference["rf_solver"]["steps"]
    scale = cfg_inference["rf_solver"]["scale"]

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
                v_pred = transformer(torch.cat([z_in, c_in], dim=1), t)
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

    print("\n" + "="*50)
    print("Final Results")
    print("="*50)
    print(f"RAW Signal CC:          {np.mean(metrics['cc']):.4f} ± {np.std(metrics['cc']):.4f}")
    print(f"Weighted Coherence:     {np.mean(metrics['coh_w']):.4f}")
    print(f"SNR (dB):               {np.mean(metrics['snr']):.2f}")
    print("-" * 30)
    print(f"ENVELOPE Correlation:   {np.mean(metrics['env_cc']):.4f} (Target Paper: ~0.63)")
    print(f"ENVELOPE Coherence:     {np.mean(metrics['env_coh']):.4f} (Target Paper: ~0.84)")
    print("="*50)

if __name__ == "__main__":
    run_evaluation()