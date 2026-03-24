import re
import random
from pathlib import Path
from src.config import Config

def evaluate_first_code_split(config):
    print("Evaluando el split del Código 1 (src/dataset.py)...\n")
    
    if not config.ECG_DIR.exists() or not config.PCG_DIR.exists():
        print("Error: No se encuentran las carpetas ECG o PCG. Revisa las rutas en tu Config.")
        return

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
    
    if total == 0:
        print("No se encontraron pares de señales.")
        return

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

    n_test = total - n_train - n_val

    # ==========================================
    # Impresión de resultados
    # ==========================================
    print("Distribución Final (Segmentos):")
    print(f"   Train: {n_train/total:.1%} ({n_train})")
    print(f"   Val:   {n_val/total:.1%} ({n_val})")
    print(f"   Test:  {n_test/total:.1%} ({n_test})")
    
    print("\nResumen del Split:")
    print(f"   Train: {len(train_subj)} sujetos | {len(train_index)} pares de señales")
    print(f"   Val:   {len(val_subj)} sujetos | {len(val_index)} pares de señales")
    print(f"   Test:  {len(test_subj)} sujetos | {len(test_index)} pares de señales")

if __name__ == "__main__":
    # Instanciamos la configuración de tu proyecto
    config = Config()
    evaluate_first_code_split(config)