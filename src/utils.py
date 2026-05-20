import random
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from src.config import Config


# Establish seed
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_comparison_plot(ecg, pcg_real, pcg_gen, path, sr = 2000):
    fig, axes = plt.subplots(3, 1, figsize(10, 8))
    t = np.linspace(0, len(ecg)/sr, len(ecg))

    axes[0].plot(t, ecg, color="black")
    axes[0].set_title("Input ECG")

    axes[1].plot(t, pcg_real, color="green", alpha=0.5, label="Real")
    axes[1].plot(t, pcg_gen, color="red", alpha = 0.8, label="Generated")
    axes[1].set_title("PCG Comparison")
    axes[1].legend()

    axes[2].specgram(pcg_gen, NFFT=1024, Fs=sr, noverlap=512, cmap="inferno")
    axes[2].set_title("Generated PCG Spectrogram")

    plt.tight_layout()
    plt.savefig(path)
    plt.close()


# Plot pair of ECG and PCG signals
def plot_single_pair():
    """Grafica el primer par de señales encontrado en el dataset."""
    if not Config.ECG_DIR.exists():
        print(f"No encuentro la carpeta: {Config.ECG_DIR}")
        return

    ecg_files = sorted(list(Config.ECG_DIR.glob("*.csv")))
    
    if not ecg_files:
        print("No hay archivos .csv en la carpeta ECG.")
        return
        
    ecg_path = ecg_files[0]
    pcg_path = Config.PCG_DIR / ecg_path.name
    
    if not pcg_path.exists():
        print(f"No encontré la pareja PCG para {ecg_path.name}")
        return

    print(f"Graficando: {ecg_path.name}")
    df_ecg = pd.read_csv(ecg_path)
    df_pcg = pd.read_csv(pcg_path)

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(12, 6), sharex=True)

    ax1.plot(df_ecg['t'], df_ecg['signal'], color='black', linewidth=1)
    ax1.set_title(f"ECG | {ecg_path.name}", loc='left', fontweight='bold')
    ax1.set_ylabel("Amp. Normalizada")
    ax1.grid(True, alpha=0.3)

    ax2.plot(df_pcg['t'], df_pcg['signal'], color='#d62728', linewidth=0.8)
    ax2.set_title("PCG", loc='left', fontstyle='italic')
    ax2.set_ylabel("Amp. Normalizada")
    ax2.set_xlabel("Tiempo (segundos)")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    
    plt.savefig("outputs/check_dataset.png")
    print("Gráfico guardado en 'outputs/check_dataset.png'")
    plt.show()

# VAE PCG Reconstruction Plot
def visualize_vae_reconstruction(model, loader, device, n_samples=1, save_path="outputs/vae_eval.png"):
    model.eval()
    with torch.no_grad():
        batch = next(iter(loader))
        real_pcg = batch["pcg"].to(device)
        recon_pcg, _, _ = model(real_pcg)
        
    real_pcg = real_pcg.cpu().numpy()
    recon_pcg = recon_pcg.cpu().numpy()

    num_samples = real_pcg.shape[-1]
    time_sec = np.arange(num_samples) / 2000
    
    fig, axes = plt.subplots(nrows=n_samples, ncols=1, figsize=(12, 3*n_samples), sharex=True)    
    
    for i in range(n_samples):
        ax = axes[i] if n_samples > 1 else axes
        ax.plot(time_sec, real_pcg[i, 0, :], color='black', alpha=0.5, linewidth=1, label="Original")
        ax.plot(time_sec, recon_pcg[i, 0, :], color='red', alpha=0.7, linewidth=1, linestyle='--', label="Reconstructed")
        ax.margins(x=0)
        ax.set_xlabel("Time (s)", fontsize=16)
        ax.set_ylabel("Amplitude", fontsize=16)
        ax.grid(True, alpha=0.8)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.legend(loc="upper right", fontsize=14)

    plt.tight_layout()
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    print(f" Gráfico de evaluación guardado en: {save_path}")
    plt.show()