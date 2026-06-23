"""
Step 4: Train student model.

Runs TWO training experiments:
  - baseline  : standard KD only          (alpha, beta > 0, gamma = 0)
  - proposed  : transition-aware KD       (alpha, beta, gamma > 0)

Both use identical architecture and hyperparameters except gamma.
Results saved to checkpoints/ and outputs/ for evaluation in step 5.
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import cohen_kappa_score, f1_score, accuracy_score
from tqdm import tqdm
import json

class SleepStudentNet(nn.Module):
    """
    Lightweight CNN-GRU student for edge deployment.
    Input  : single EEG epoch (1, 3000)
    Output : 5-class logits
    """
    def __init__(self, n_classes=5, gru_hidden=64, dropout=0.3):
        super().__init__()

        self.cnn = nn.Sequential(
            # Layer 1: large receptive field for slow waves
            nn.Conv1d(1, 16, kernel_size=50, stride=5, padding=25),
            nn.BatchNorm1d(16), nn.ReLU(),
            nn.MaxPool1d(2),

            # Layer 2: mid-range features
            nn.Conv1d(16, 32, kernel_size=10, stride=2, padding=5),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.MaxPool1d(2),

            # Layer 3: fine features
            nn.Conv1d(32, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.AdaptiveAvgPool1d(32),   # fixed output length
        )

        self.dropout = nn.Dropout(dropout)
        self.gru     = nn.GRU(32, gru_hidden, batch_first=True, num_layers=1)
        self.fc      = nn.Linear(gru_hidden, n_classes)

    def forward(self, x, h=None):
        """
        x : (B, 3000)  single epoch per sample
        h : GRU hidden state — pass across consecutive epochs for temporal context
        Returns: logits (B, n_classes), h_new
        """
        x = x.unsqueeze(1)                    # (B, 1, 3000)
        feat = self.cnn(x)                    # (B, 32, 32)
        feat = feat.permute(0, 2, 1)          # (B, 32, 32) — seq for GRU
        feat = self.dropout(feat)
        out, h_new = self.gru(feat, h)        # (B, 32, 64)
        logits = self.fc(out[:, -1, :])       # last timestep → (B, 5)
        return logits, h_new

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class TransitionDataset(Dataset):
    """
    Loads the precomputed transition dataset from step 3.
    Each item is a consecutive epoch pair with teacher soft labels.
    """
    def __init__(self, npz_path, indices=None):
        d = np.load(npz_path)
        self.x_t  = d['x_t']    # (N, 3000)
        self.x_t1 = d['x_t1']   # (N, 3000)
        self.p_t  = d['p_t']    # (N, 5)  teacher soft at t
        self.p_t1 = d['p_t1']   # (N, 5)  teacher soft at t+1
        self.y_t  = d['y_t']    # (N,)    hard label at t
        self.w_t  = d['w_t']    # (N,)    confidence weight

        if indices is not None:
            self.x_t  = self.x_t[indices]
            self.x_t1 = self.x_t1[indices]
            self.p_t  = self.p_t[indices]
            self.p_t1 = self.p_t1[indices]
            self.y_t  = self.y_t[indices]
            self.w_t  = self.w_t[indices]

    def __len__(self):
        return len(self.y_t)

    def __getitem__(self, idx):
        return {
            'x_t':  torch.FloatTensor(self.x_t[idx]),
            'x_t1': torch.FloatTensor(self.x_t1[idx]),
            'p_t':  torch.FloatTensor(self.p_t[idx]),
            'p_t1': torch.FloatTensor(self.p_t1[idx]),
            'y_t':  torch.LongTensor([self.y_t[idx]])[0],
            'w_t':  torch.FloatTensor([self.w_t[idx]])[0],
        }
def compute_loss(student, batch, device, alpha, beta, gamma, tau,
                 return_components=False):
    x_t  = batch['x_t'].to(device)    # (B, 3000)
    x_t1 = batch['x_t1'].to(device)   # (B, 3000)
    p_t  = batch['p_t'].to(device)    # (B, 5)
    p_t1 = batch['p_t1'].to(device)   # (B, 5)
    y_t  = batch['y_t'].to(device)    # (B,)
    w_t  = batch['w_t'].to(device)    # (B,)

    # --- Forward pass (pass hidden state t → t+1 for temporal coupling) ---
    logits_t,  h = student(x_t)
    logits_t1, _ = student(x_t1, h.detach())   # detach to avoid double backprop

    # Term 1: Cross-entropy on hard labels 
    L_ce = F.cross_entropy(logits_t, y_t)

    # Term 2: Standard KD — match teacher soft labels at t 
    log_q_t  = F.log_softmax(logits_t / tau, dim=-1)
    soft_p_t = F.softmax(p_t / tau, dim=-1)
    L_kd = F.kl_div(log_q_t, soft_p_t, reduction='batchmean') * (tau ** 2)

    # Term 3: Transition loss — student at t+1 matches teacher at t+1 w teacher confidence
    log_q_t1  = F.log_softmax(logits_t1 / tau, dim=-1)
    soft_p_t1 = F.softmax(p_t1 / tau, dim=-1)
    kl_per    = F.kl_div(log_q_t1, soft_p_t1, reduction='none').sum(dim=-1)  # (B,)
    L_trans   = (w_t * kl_per).mean() * (tau ** 2)

    L_total = alpha * L_ce + beta * L_kd + gamma * L_trans

    if return_components:
        return L_total, {
            'ce':    L_ce.item(),
            'kd':    L_kd.item(),
            'trans': L_trans.item(),
            'total': L_total.item(),
        }
    return L_total

def train_student(config, train_dl, val_dl, device, save_path):
    model = SleepStudentNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'],
                                 weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['epochs'])

    alpha = config['alpha']
    beta  = config['beta']
    gamma = config['gamma']
    tau   = config['tau']

    history = {
        'train_loss': [], 'val_loss': [],
        'val_acc': [], 'val_kappa': [], 'val_f1': [],
        'loss_ce': [], 'loss_kd': [], 'loss_trans': [],
    }
    best_f1 = -1.0

    print(f"Training: {config['name']}")
    print(f"  alpha={alpha}, beta={beta}, gamma={gamma}, tau={tau}")
    print(f"  Params: {model.count_params():,}")

    for ep in range(1, config['epochs'] + 1):
        model.train()
        ep_loss = {'ce': 0, 'kd': 0, 'trans': 0, 'total': 0}
        n_batches = 0

        for batch in tqdm(train_dl, desc=f"Ep {ep}/{config['epochs']}", leave=False):
            optimizer.zero_grad()
            loss, comps = compute_loss(model, batch, device,
                                       alpha, beta, gamma, tau,
                                       return_components=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            for k in ep_loss:
                ep_loss[k] += comps.get(k, 0)
            n_batches += 1

        scheduler.step()

        for k in ep_loss:
            ep_loss[k] /= n_batches

        model.eval()
        all_preds, all_true = [], []
        val_loss_sum = 0

        with torch.no_grad():
            for batch in val_dl:
                loss, _ = compute_loss(model, batch, device,
                                       alpha, beta, gamma, tau,
                                       return_components=True)
                val_loss_sum += loss.item()
                logits, _ = model(batch['x_t'].to(device))
                preds = logits.argmax(dim=-1).cpu().numpy()
                true  = batch['y_t'].numpy()
                all_preds.extend(preds)
                all_true.extend(true)

        all_preds = np.array(all_preds)
        all_true  = np.array(all_true)
        val_acc   = accuracy_score(all_true, all_preds)
        val_kappa = cohen_kappa_score(all_true, all_preds)
        val_f1    = f1_score(all_true, all_preds, average='macro',
                             zero_division=0)
        val_loss  = val_loss_sum / len(val_dl)

        history['train_loss'].append(ep_loss['total'])
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_kappa'].append(val_kappa)
        history['val_f1'].append(val_f1)
        history['loss_ce'].append(ep_loss['ce'])
        history['loss_kd'].append(ep_loss['kd'])
        history['loss_trans'].append(ep_loss['trans'])

        print(f"  Ep {ep:3d} | loss {ep_loss['total']:.4f} "
              f"[ce={ep_loss['ce']:.3f} kd={ep_loss['kd']:.3f} "
              f"tr={ep_loss['trans']:.3f}] | "
              f"val acc={val_acc:.3f} κ={val_kappa:.3f} F1={val_f1:.3f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), save_path)

    print(f"\nBest val F1: {best_f1:.4f} — saved to {save_path}")
    model.load_state_dict(torch.load(save_path, map_location=device))
    return model, history


def benchmark_speed(model, device, n_runs=500):
    """Measure inference time per epoch in milliseconds."""
    model.eval()
    dummy = torch.randn(1, 3000).to(device)

    # Warmup
    for _ in range(50):
        with torch.no_grad():
            _ = model(dummy)

    # Timed runs
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            model(dummy)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)   # ms

    return {
        'mean_ms':   float(np.mean(times)),
        'std_ms':    float(np.std(times)),
        'min_ms':    float(np.min(times)),
        'median_ms': float(np.median(times)),
    }

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("outputs",     exist_ok=True)

    # Load dataset
    dataset_path = "outputs/transition_dataset.npz"

    from split_utils import subject_level_split, verify_no_leakage

    raw = np.load(dataset_path)
    subject_ids = raw['subject_id']
    N = len(subject_ids)

    train_idx, val_idx, train_subj, val_subj = subject_level_split(
        subject_ids, val_frac=0.2, seed=42)

    # Verify zero leakage before proceeding
    leak_frac = verify_no_leakage(raw['x_t'], raw['x_t1'], train_idx, val_idx)
    if leak_frac > 0.001:
        print("Leakage detected even after subject-level split. "
              "Check split_utils.py logic before continuing.")
        return

    train_ds = TransitionDataset(dataset_path, indices=train_idx)
    val_ds   = TransitionDataset(dataset_path, indices=val_idx)

    train_dl = DataLoader(train_ds, batch_size=256, shuffle=True,
                          num_workers=0, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=256, shuffle=False,
                          num_workers=0, pin_memory=True)

    print(f"Dataset: {len(train_idx):,} train pairs | {len(val_idx):,} val pairs")

    configs = [
        {
            'name':   'baseline_kd',
            'alpha':  0.3,
            'beta':   0.7,
            'gamma':  0.0,    
            'tau':    2.0,
            'lr':     1e-3,
            'epochs': 40,
        },
        {
            'name':   'proposed_tad',   # Transition-Aware Distillation
            'alpha':  0.3,
            'beta':   0.5,
            'gamma':  0.2,    
            'tau':    2.0,
            'lr':     1e-3,
            'epochs': 40,
        },
    ]

    results = {}

    for cfg in configs:
        save_path = f"checkpoints/student_{cfg['name']}.pt"
        model, history = train_student(cfg, train_dl, val_dl, device, save_path)

        # Final evaluation on val set
        model.eval()
        all_preds, all_true, all_probs = [], [], []
        t_start = time.perf_counter()

        with torch.no_grad():
            for batch in val_dl:
                logits, _ = model(batch['x_t'].to(device))
                probs = F.softmax(logits, dim=-1).cpu().numpy()
                preds = logits.argmax(dim=-1).cpu().numpy()
                all_preds.extend(preds)
                all_true.extend(batch['y_t'].numpy())
                all_probs.extend(probs)

        inference_time = (time.perf_counter() - t_start) / len(all_preds) * 1000

        all_preds = np.array(all_preds)
        all_true  = np.array(all_true)
        all_probs = np.array(all_probs)

        # Per-stage F1
        stage_f1 = f1_score(all_true, all_preds, average=None,
                             labels=[0,1,2,3,4], zero_division=0)

        speed = benchmark_speed(model, device)

        results[cfg['name']] = {
            'config':      cfg,
            'history':     history,
            'params':      model.count_params(),
            'accuracy':    float(accuracy_score(all_true, all_preds)),
            'kappa':       float(cohen_kappa_score(all_true, all_preds)),
            'macro_f1':    float(f1_score(all_true, all_preds, average='macro',
                                          zero_division=0)),
            'stage_f1':    stage_f1.tolist(),
            'speed_ms':    speed,
            'predictions': all_preds.tolist(),
            'true_labels': all_true.tolist(),
            'probs':       all_probs.tolist(),
        }

        print(f"\n[{cfg['name']}] Final metrics:")
        print(f"  Accuracy : {results[cfg['name']]['accuracy']:.4f}")
        print(f"  Kappa    : {results[cfg['name']]['kappa']:.4f}")
        print(f"  Macro F1 : {results[cfg['name']]['macro_f1']:.4f}")
        print(f"  Per-stage F1: " +
              " | ".join([f"{s}={v:.3f}" for s, v in
                          zip(["W","N1","N2","N3","REM"], stage_f1)]))
        print(f"  Inference: {speed['mean_ms']:.3f} ms/epoch "
              f"(±{speed['std_ms']:.3f})")
        print(f"  Params   : {model.count_params():,}")

    # Save results
    # Convert numpy arrays to lists for JSON serialization
    def to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, dict):
            return {k: to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [to_serializable(i) for i in obj]
        return obj

    with open("outputs/training_results.json", "w") as f:
        json.dump(to_serializable(results), f, indent=2)


if __name__ == "__main__":
    main()