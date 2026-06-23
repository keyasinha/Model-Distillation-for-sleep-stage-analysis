import os
import json
import glob
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import cohen_kappa_score, f1_score, accuracy_score
from tqdm import tqdm

from split_utils import subject_level_split

STAGE_NAMES = ["Wake", "N1", "N2", "N3", "REM"]
SEQ_LEN     = 20
EPOCH_SAMP  = 3000

def load_teacher(ckpt_path, device):
    import sys
    sys.path.insert(0, ".")
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "teacher_module", "2_teacher_inference.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    model = mod.TinySleepNet().to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    return model


def evaluate_teacher_on_subjects(teacher, subject_ids, data_dir, device):
    """Run teacher on specific subjects, return per-epoch preds + labels."""
    all_preds, all_true = [], []

    for subj_id in tqdm(subject_ids, desc="Evaluating teacher"):
        npz_path = os.path.join(data_dir, f"{subj_id}.npz")
        if not os.path.exists(npz_path):
            continue

        d = np.load(npz_path)
        X, y = d['x'], d['y']   # (N, 3000), (N,)
        N = len(X)

        with torch.no_grad():
            idx = 0
            while idx < N:
                end = min(idx + SEQ_LEN, N)
                chunk = X[idx:end]
                if len(chunk) < SEQ_LEN:
                    pad = np.zeros((SEQ_LEN - len(chunk), EPOCH_SAMP),
                                   dtype=np.float32)
                    chunk = np.concatenate([chunk, pad], axis=0)

                x_t = torch.FloatTensor(chunk).unsqueeze(0).to(device)
                logits, _ = teacher(x_t)
                preds = logits[0].argmax(-1).cpu().numpy()

                actual_len = end - idx
                all_preds.extend(preds[:actual_len].tolist())
                all_true.extend(y[idx:end].tolist())
                idx = end

    return np.array(all_preds), np.array(all_true)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt_path = "checkpoints/teacher.pt"
    if not os.path.exists(ckpt_path):
        print(f"Teacher checkpoint not found: {ckpt_path}")
        return

    dataset_path = "outputs/transition_dataset.npz"
    if not os.path.exists(dataset_path):
        print("Transition dataset not found.")
        return

    raw = np.load(dataset_path)
    subject_ids_all = raw['subject_id']

    # Exact same split as all student experiments
    _, _, _, val_subjects = subject_level_split(
        subject_ids_all, val_frac=0.2, seed=42)
    val_subjects = sorted(val_subjects)
    print(f"Evaluating teacher on val subjects: {val_subjects}")

    print("Loading teacher checkpoint")
    teacher = load_teacher(ckpt_path, device)

    data_dir = "data/processed"
    preds, true = evaluate_teacher_on_subjects(
        teacher, val_subjects, data_dir, device)

    acc      = float(accuracy_score(true, preds))
    kappa    = float(cohen_kappa_score(true, preds))
    macro_f1 = float(f1_score(true, preds, average='macro', zero_division=0))
    stage_f1 = f1_score(true, preds, average=None,
                        labels=[0,1,2,3,4], zero_division=0).tolist()

    metrics = {
        'accuracy':  acc,
        'kappa':     kappa,
        'macro_f1':  macro_f1,
        'stage_f1':  stage_f1,
        'val_subjects': val_subjects,
        'n_epochs':  len(true),
    }

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/teacher_val_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    
    print("TEACHER METRICS (same val split as students)")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Kappa    : {kappa:.4f}")
    print(f"  Macro F1 : {macro_f1:.4f}")
    print("  Per-stage F1: " + " | ".join(
        [f"{s}={v:.3f}" for s,v in zip(STAGE_NAMES, stage_f1)]))
    print(f"\nSaved: outputs/teacher_val_metrics.json")


if __name__ == "__main__":
    main()