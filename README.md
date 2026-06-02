# ECG-to-PCG

---

## 📂 Repository Structure

The repository is organized as follows:

```text
ECG-to-PCG/
│
├── configs/                # Configuration files and experimental parameters
├── checkpoints/            # Trained models and saved weights
├── diagram_assets/         # Images and diagrams for documentation
├── src/                    # Base source code (data loaders, models, utilities)
│
├── outputs/                # Main generation results
├── cfg_scale_results/      # Experimental results when varying the CFG scale parameter
├── cfg_study_results/      # Plots and analysis of the CFG studies
│
├── train_vae.py            # Script to train the Variational Autoencoder (VAE)
├── train_flow.py           # Script to train the complete flow model
├── generate.py             # Inference script to generate PCG from ECG
├── eval_vae.py             # Script for evaluating the VAE model (reconstruction)
├── test.py                 # Additional tests and validations
└── proyecto_info.txt       # Additional project information
```
## ⚠️ Data Preparation (CRITICAL)

Due to GitHub storage limitations, the data folder is not included in this repository. For the code to work correctly, you must download the data and manually organize the structure.

The dataset used is EPHNOGRAM.
The signals must meet the following preprocessing requirements:
- Sampling rate (Downsampled): 2000 Hz
- Segment length: 5 seconds
### Required Directory Structure
You must create the following structure in the root directory of the project:
```text
ECG-to-PCG/
└── data/
    └── dataset_downsampled/
        ├── ECG/
        │   ├── ECGPCG_0001_seg0001.csv
        │   ├── ECGPCG_0001_seg0002.csv
        │   └── ... (all ECG .csv files)
        └── PCG/
            ├── ECGPCG_0001_seg0001.csv
            ├── ECGPCG_0001_seg0002.csv
            └── ... (all PCG .csv files)
```
*(Ensure that the filenames match exactly in both the ECG and PCG folders for proper sample pairing).*
## 🚀 Execution Order and Usage

Once you have the dependencies installed and the data folder correctly placed, follow this main execution order to reproduce the project:
### 1. Configuration

Before running the scripts, check the configs/ folder to adjust training hyperparameters, paths, and model architectural features if necessary.
### 2. Train the VAE

First, you need to train the Variational Autoencoder (VAE) to learn the latent representations of the signals. Run:
```code
python train_vae.py
```
*(You can evaluate the quality of its reconstructions using python eval_vae.py once trained).*

### 3. Train the Flow Model

After the VAE is successfully trained, train the complete flow-based model by running:
```code
python train_flow.py
```
*Model weights and checkpoints will be saved automatically in the checkpoints/ folder as training progresses.*

### 4. Signal Generation (Inference)

Once the full model is trained, you can generate synthetic PCG signals conditioned on ECG signals using:
```code
python generate.py
```
*This script will take validation/test ECG samples and generate their corresponding PCG files. The results will be saved in outputs/.*

### 5. Classifier-Free Guidance (CFG) Study (Optional)

The repository contains support for parametric studies on CFG. Generations with different guidance scales can be analyzed by checking the directories:

- cfg_scale_results/: Contains files generated with different levels of CFG.

- cfg_study_results/: Contains plots or metrics comparing the impact of CFG on the quality and fidelity of the generated signal.
