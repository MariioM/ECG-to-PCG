import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Sample frequency for ECG and PCG

fs_ecg = 2000
fs_pcg = 2000

base_dir = os.path.join("data", "dataset_downsampled")
ecg_file = os.path.join(base_dir, "ECG", "ECGPCG0001__seg0002.csv")
pcg_file = os.path.join(base_dir, "PCG", "ECGPCG0001__seg0002.csv")

try:
    df_ecg = pd.read_csv(ecg_file)
    df_pcg = pd.read_csv(pcg_file)
    
    ecg_data = df_ecg['signal'].values
    pcg_data = df_pcg['signal'].values
except FileNotFoundError as e:
    print(f"Error: {e}")
    exit()
except KeyError as e:
    print(f"Error: {e}")
    exit()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)

nperseg_val = 1024
noverlap_val = nperseg_val // 2

f_ecg, t_ecg, Sxx_ecg = signal.spectrogram(ecg_data, fs_ecg, window='hann', nperseg=nperseg_val, noverlap=noverlap_val)
Sxx_ecg_db = 10 * np.log10(Sxx_ecg + 1e-10)

mesh1 = ax1.pcolormesh(t_ecg, f_ecg, Sxx_ecg_db, shading='gouraud', cmap='magma')
ax1.set_ylabel("Frecuencia (Hz)")
ax1.set_ylim(0, 1000)
ax1.axhline(0.05, color='cyan', linestyle='--', linewidth=1.5, label='Banda útil (0.05 - 180 Hz)')
ax1.axhline(180, color='cyan', linestyle='--', linewidth=1.5)
ax1.grid(True, alpha=0.8)
ax1.tick_params(axis='both', which='major', labelsize=14)
ax1.legend(loc='upper right')
ax1.margins(x=0)
fig.colorbar(mesh1, ax=ax1, label='Potencia (dB)')

f_pcg, t_pcg, Sxx_pcg = signal.spectrogram(pcg_data, fs_pcg, window='hann', nperseg=nperseg_val, noverlap=noverlap_val)
Sxx_pcg_db = 10 * np.log10(Sxx_pcg + 1e-10)

mesh2 = ax2.pcolormesh(t_pcg, f_pcg, Sxx_pcg_db, shading='gouraud', cmap='magma')
ax2.set_title("Espectrograma del PCG", fontsize=12, fontweight='bold')
ax2.set_xlabel("Tiempo (s)")
ax2.set_ylabel("Frecuencia (Hz)")
ax2.set_ylim(0, 1000)
ax2.axhline(20, color='cyan', linestyle='--', linewidth=1.5, label='Banda útil (20 - 800 Hz)')
ax2.axhline(800, color='cyan', linestyle='--', linewidth=1.5)
ax2.grid(True, alpha=0.8)
ax2.tick_params(axis='both', which='major', labelsize=14)
ax2.legend(loc='upper right')
ax2.margins(x=0)
fig.colorbar(mesh2, ax=ax2, label='Potencia (dB)')

os.makedirs("plots", exist_ok=True)
plt.savefig(os.path.join("plots", "Espectrogramas_ECG_PCG.png"), format='png', dpi=300)
plt.show()