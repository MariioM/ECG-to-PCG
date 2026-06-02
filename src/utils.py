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

# Comparison plot between real and sintetic PCG.
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


# PLOT A PAIR OF ECG AND PCG FROM DATASET
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

# VAE PCG RECONSTRUCTION PLOT
def visualize_vae_reconstruction(model, loader, device, n_samples=4, save_path="outputs/vae_eval.png"):
    model.eval()
    with torch.no_grad():
        batch = next(iter(loader))
        real_pcg = batch["pcg"].to(device)
        recon_pcg, _, _ = model(real_pcg)
        
    real_pcg = real_pcg.cpu().numpy()
    recon_pcg = recon_pcg.cpu().numpy()
    
    fig, axes = plt.subplots(nrows=n_samples, ncols=1, figsize=(12, 3*n_samples), sharex=True)
    fig.suptitle("Evaluación VAE: Original (Negro) vs Reconstrucción (Rojo)", fontsize=14, y=1.0)
    
    for i in range(n_samples):
        ax = axes[i] if n_samples > 1 else axes
        ax.plot(real_pcg[i, 0, :], color='black', alpha=0.5, label='Original', linewidth=1)
        ax.plot(recon_pcg[i, 0, :], color='red', alpha=0.7, label='Reconstrucción', linewidth=1, linestyle='--')
        ax.set_title(f"Muestra #{i+1}", loc='left', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.2)
        if i == 0: ax.legend(loc="upper right")
        
    axes[-1].set_xlabel("Muestras")
    plt.tight_layout()
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    print(f" Gráfico de evaluación guardado en: {save_path}")
    plt.show()
