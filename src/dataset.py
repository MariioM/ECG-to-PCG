# src/dataset.py
import re
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
from src.config import Config

def load_csv_1d(path: str) -> np.ndarray:
    try:
        arr = np.genfromtxt(path, delimiter=",", skip_header=1)
    except Exception:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            first = f.readline().strip()
        has_header = bool(re.search(r"[A-Za-z]", first))
        delim = "," if first.count(",") >= first.count(";") else ";"
        arr = np.genfromtxt(path, delimiter=delim, skip_header=1 if has_header else 0)

    if arr.ndim == 0:
        raise ValueError(f"Archivo vacío o ilegible: {path}")

    x = arr.astype(np.float32) if arr.ndim == 1 else arr[:, -1].astype(np.float32)

    if not np.isfinite(x).all():
        x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32)

    return x

def fix_length(x: np.ndarray, target_len: int) -> np.ndarray:
    n = x.shape[0]
    if n == target_len:
        return x
    if n > target_len:
        start = (n - target_len) // 2
        return x[start:start + target_len]
    pad = target_len - n
    return np.pad(x, (0, pad), mode="constant")

class ECGPCGDataset(Dataset):
    def __init__(self, index_pairs, seq_len, cache_name="train"):
        self.index = index_pairs
        self.seq_len = seq_len
        
        cache_file = Path(f"dataset_cache_{cache_name}.pt")

        if cache_file.exists():
            print(f" Cargando caché desde disco ({cache_file})...")
            checkpoint = torch.load(cache_file)
            self.cached_ecg = checkpoint['ecg']
            self.cached_pcg = checkpoint['pcg']
            self.cached_paths = checkpoint['paths']
        else:
            self.cached_ecg = []
            self.cached_pcg = []
            self.cached_paths = []
            
            print(f" Cargando {len(index_pairs)} señales en RAM y creando caché...")
            for ecg_path, pcg_path in tqdm(index_pairs, desc=f"Caching {cache_name}"):
                sig_ecg = fix_length(load_csv_1d(ecg_path), self.seq_len)
                sig_pcg = fix_length(load_csv_1d(pcg_path), self.seq_len)
                
                self.cached_ecg.append(torch.from_numpy(sig_ecg).unsqueeze(0).float())
                self.cached_pcg.append(torch.from_numpy(sig_pcg).unsqueeze(0).float())
                self.cached_paths.append((str(ecg_path), str(pcg_path)))
            
            # Guardamos en disco para la próxima vez
            torch.save({
                'ecg': self.cached_ecg,
                'pcg': self.cached_pcg,
                'paths': self.cached_paths
            }, cache_file)
            print(f" Caché guardada en {cache_file}")

    def __len__(self):
        return len(self.cached_ecg)

    def __getitem__(self, i):
        return {
            "ecg": self.cached_ecg[i],
            "pcg": self.cached_pcg[i],
            "path_ecg": self.cached_paths[i][0],
            "path_pcg": self.cached_paths[i][1]
        }

def get_dataloaders(config):
    ecg_files = sorted([p for p in config.ECG_DIR.glob("*.csv")])
    all_pairs = []
    for ecg_path in ecg_files:
        pcg_path = config.PCG_DIR / ecg_path.name
        if pcg_path.exists():
            all_pairs.append((ecg_path, pcg_path))

    subject_groups = {}
    regex_subj = re.compile(r"^(.+)__seg\d+")

    for ecg_p, pcg_p in all_pairs:
        match = regex_subj.match(ecg_p.name)
        if not match: continue
        subject_id = match.group(1)
        subject_groups.setdefault(subject_id, []).append((ecg_p, pcg_p))

    subjects = sorted(list(subject_groups.keys()))
    random.Random(config.SEED).shuffle(subjects)

    counts = {s: len(subject_groups[s]) for s in subjects}
    total = sum(counts.values())
    target_train = total * 0.7  
    target_val = total * 0.2

    train_subj, val_subj, test_subj = [], [], []
    n_train = n_val = 0

    for s in subjects:
        n = counts[s]
        if n_train + n <= target_train:
            train_subj.append(s)
            n_train += n
        elif n_val + n <= target_val:
            val_subj.append(s)
            n_val += n
        else:
            test_subj.append(s)

    train_index = sum([subject_groups[s] for s in train_subj], [])
    val_index = sum([subject_groups[s] for s in val_subj], [])
    test_index = sum([subject_groups[s] for s in test_subj], [])

    train_ds = ECGPCGDataset(train_index, config.SEQ_LEN, cache_name="train")
    val_ds = ECGPCGDataset(val_index, config.SEQ_LEN, cache_name="val")
    test_ds = ECGPCGDataset(test_index, config.SEQ_LEN, cache_name="test")

    train_loader = DataLoader(
        train_ds, batch_size=config.BATCH_SIZE, shuffle=True, 
        num_workers=0, pin_memory=config.PIN_MEMORY, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.BATCH_SIZE, shuffle=False, 
        num_workers=0, pin_memory=config.PIN_MEMORY, drop_last=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=config.BATCH_SIZE, shuffle=False, 
        num_workers=0, pin_memory=config.PIN_MEMORY
    )

    return train_loader, val_loader, test_loader