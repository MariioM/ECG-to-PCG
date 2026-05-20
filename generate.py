import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.config import Config
from src.models import PCGVAE, FlowTransformer, Encoder1D
from src.dataset import get_dataloaders
from src.utils import seed_everything

from scipy import signal

def generate_pcg():
    seed_everything(Config.SEED)
    cfg_models = Config.load_model_config()
    cfg_inference = Config.load_inference_config()
    _,_, test_loader = get_dataloaders(Config)

    device = Config.DEVICE
    print("=======================================================")
    print(device)
    print("=======================================================")

    
    # Instance VAE
    vae = PCGVAE(
        in_channels=cfg_models["vae"]["in_channels"],
        base_channels=cfg_models["vae"]["base_channels"],
        z_dim=cfg_models["vae"]["z_dim"]
    ).to(device)
    vae.load_state_dict(torch.load(Config.CHECK_DIR / cfg_models["paths"]["vae_checkpoint"]))
    vae.eval()

    # Load ECG Encoder and Transformer
    ecg_encoder = Encoder1D(in_channels=1, base_channels=32, z_dim=16).to(device)
    transformer = FlowTransformer(in_channels=16, ecg_channels=16,
                hidden_size=cfg_models["rf"]["hidden_size"], 
                depth=cfg_models["rf"]["depth"], 
                num_heads=cfg_models["rf"]["num_heads"]).to(device)

    flow_ckpt_path = Config.CHECK_DIR / cfg_models['paths']['flow_checkpoint']
    checkpoint = torch.load(flow_ckpt_path, map_location=device)

    ecg_encoder.load_state_dict(checkpoint['ecg_encoder'])
    transformer.load_state_dict(checkpoint['transformer'])

    # Recover normalization stats
    STATS_MEAN = checkpoint['norm_stats']['mean'].to(device)
    STATS_STD = checkpoint['norm_stats']['std'].to(device)

    ecg_encoder.eval()
    transformer.eval()

    # Obtain test data
    batch = next(iter(test_loader))
    real_ecg = batch["ecg"].to(device)
    real_pcg = batch["pcg"].to(device)

    # CFG Setup
    STEPS = cfg_inference["rf_solver"]["steps"]
    SCALE = cfg_inference["rf_solver"]["scale"]
    B = real_ecg.shape[0]
    with torch.no_grad():
        # Setup conditions
        cond, _ = ecg_encoder(real_ecg)
        uncond = torch.zeros_like(cond).to(device)

        # Begin from noise
        z = torch.randn(B, 16, 312).to(device)
        dt = 1.0 / STEPS

        for i in range(STEPS):
            t_val = i / STEPS

            z_doubled = torch.cat([z, z], dim = 0)
            cond_doubled = torch.cat([cond, uncond], dim=0)
            t_doubled = torch.full((B * 2,), t_val, device=device)

            v_pred_doubled = transformer(x=z_doubled, t=t_doubled, c_ecg=cond_doubled)

            v_cond, v_uncond = v_pred_doubled.chunk(2, dim=0)

            v_final = v_uncond +  SCALE * (v_cond - v_uncond)

            # Euler Step
            z = z + v_final * dt

        z_final = (z * STATS_STD) + STATS_MEAN
        pcg_fake = vae.decode(z_final, target_len=Config.SEQ_LEN)

    # Plot the results
    idx = 1
    t_axis = np.linspace(0, 5, Config.SEQ_LEN)
    sig_ecg = real_ecg[idx, 0, :].cpu().numpy()
    sig_real = real_pcg[idx, 0, :].cpu().numpy()
    
    sig_fake_raw = pcg_fake[idx, 0, :].cpu().numpy()

    sig_fake_filtered = apply_lowpass_filter(sig_fake_raw, cutoff=400, fs=2000)

    def save_comparison_plot(sig_fake_to_plot, filename, title_suffix):
        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1])

        ax0 = fig.add_subplot(gs[0, :])
        ax0.plot(t_axis, sig_ecg, color='black')
        ax0.set_xlabel("Time (s)", fontsize=16)
        ax0.set_ylabel("Amplitude", fontsize=16)
        ax0.legend(loc="upper right")

        ax1 = fig.add_subplot(gs[1, 0])
        ax1.plot(t_axis, sig_real, color='green', alpha=0.7)
        ax1.set_xlabel("Time (s)", fontsize=16)
        ax1.set_ylabel("Amplitude", fontsize=16)
        ax1.legend(loc="upper right")

        ax2 = fig.add_subplot(gs[1, 1], sharey=ax1)
        ax2.plot(t_axis, sig_fake_to_plot, color='red', alpha=0.8)
        ax2.set_xlabel("Time (s)", fontsize=16)
        ax2.set_ylabel("Amplitude", fontsize=16)
        ax2.legend(loc="upper right")

        ax3 = fig.add_subplot(gs[2, 0])
        ax3.specgram(sig_real, NFFT=256, Fs=2000, noverlap=128, cmap='magma')
        ax3.set_xlabel("Time (s)", fontsize=16)
        ax3.set_ylabel("Frequency (Hz)", fontsize=16)

        ax4 = fig.add_subplot(gs[2, 1])
        ax4.specgram(sig_fake_to_plot, NFFT=256, Fs=2000, noverlap=128, cmap='magma')
        ax4.set_xlabel("Time (s)", fontsize=16)
        ax4.set_ylabel("Frequency (Hz)", fontsize=16)

        plt.tight_layout()
        plt.savefig(f"outputs/{filename}")
        plt.close()
        print(f"Plot saved: outputs/{filename}")

    def save_ecg_plot(filename="ecg_only.png"):
        fig, ax = plt.subplots(figsize=(15, 3))
        ax.plot(t_axis, sig_ecg, color='black')
        ax.set_xlabel("Time (s)", fontsize=16)
        ax.set_ylabel("Amplitude", fontsize=16)
        ax.grid(True, alpha=0.8)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.margins(x=0)
        ax.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(f"outputs/{filename}")
        plt.close()
        print(f"Plot saved: outputs/{filename}")

    def save_pcg_only_plot(sig_fake_to_plot, filename, title_suffix):
        fig = plt.figure(figsize=(15, 7))
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])

        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(t_axis, sig_real, color='green', alpha=0.7)
        ax1.set_xlabel("Time (s)", fontsize=16)
        ax1.set_ylabel("Amplitude", fontsize=16)
        ax1.grid(True, alpha=0.8)
        ax1.tick_params(axis='both', which='major', labelsize=14)
        ax1.margins(x=0)
        ax1.legend(loc="upper right")

        ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)
        ax2.plot(t_axis, sig_fake_to_plot, color='red', alpha=0.8)
        ax2.set_xlabel("Time (s)", fontsize=16)
        ax2.set_ylabel("Amplitude", fontsize=16)
        ax2.grid(True, alpha=0.8)
        ax2.tick_params(axis='both', which='major', labelsize=14)
        ax2.margins(x=0)
        ax2.legend(loc="upper right")

        ax3 = fig.add_subplot(gs[1, 0])
        _, _, _, im3 = ax3.specgram(sig_real, NFFT=256, Fs=2000, noverlap=128, cmap='magma')
        ax3.specgram(sig_real, NFFT=256, Fs=2000, noverlap=128, cmap='magma')
        ax3.set_xlabel("Time (s)", fontsize=16)
        ax3.set_ylabel("Frequency (Hz)", fontsize=16)
        ax3.tick_params(axis='both', which='major', labelsize=14)
        cbar3 = fig.colorbar(im3, ax=ax3)
        cbar3.set_label('Power (dB)', fontsize=16)
        cbar3.ax.tick_params(labelsize=14)

        ax4 = fig.add_subplot(gs[1, 1])
        _, _, _, im4 = ax4.specgram(sig_fake_to_plot, NFFT=256, Fs=2000, noverlap=128, cmap='magma')
        ax4.specgram(sig_fake_to_plot, NFFT=256, Fs=2000, noverlap=128, cmap='magma')
        ax4.set_xlabel("Time (s)", fontsize=16)
        ax4.set_ylabel("Frequency (Hz)", fontsize=16)
        ax4.tick_params(axis='both', which='major', labelsize=14)
        cbar4 = fig.colorbar(im4, ax=ax4)
        cbar4.set_label('Power (dB)', fontsize=16)
        cbar4.ax.tick_params(labelsize=14)

        plt.tight_layout()
        plt.savefig(f"outputs/{filename}")
        plt.close()
        print(f"Plot saved: outputs/{filename}")
    

    save_comparison_plot(sig_fake_raw, "generation_raw.png", "(RAW)")
    save_comparison_plot(sig_fake_filtered, "generation_filtered.png", "(FILTERED)")

    save_ecg_plot("generation_ecg_only.png")

    save_pcg_only_plot(sig_fake_raw, "generation_pcg_only_raw.png", "(RAW)")
    save_pcg_only_plot(sig_fake_filtered, "generation_pcg_only_filtered.png", "(FILTERED)")


def apply_lowpass_filter(data, cutoff=400, fs=2000, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    return signal.filtfilt(b, a, data)

if __name__ == "__main__":

    Path("outputs").mkdir(parents=True, exist_ok=True)
    generate_pcg()