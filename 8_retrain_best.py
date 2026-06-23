"""
Step 8: Retrain student with best sweep config and regenerate all evaluation plots.

Reads best_config from outputs/sweep_results.json, trains a fresh student
for 60 epochs (longer than sweep runs), saves predictions in the same format
as step 4, then calls all plots from step 5.

Run after 6_sweep.py and 7_sweep_plots.py.
"""

import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import (cohen_kappa_score, f1_score, accuracy_score)
from tqdm import tqdm

STAGE_NAMES = ["Wake", "N1", "N2", "N3", "REM"]

# -------------------------------------------------------------------
# Copy architecture + dataset (same as steps 4 and 6)
# -------------------------------------------------------------------
class SleepStudentNet(nn.Module):
    def __init__(self, n_classes=5, gru_hidden=64, dropout=0.3):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=50, stride=5, padding=25),
            nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=10, stride=2, padding=5),
            nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.AdaptiveAvgPool1d(32),
        )
        self.dropout = nn.Dropout(dropout)
        self.gru     = nn.GRU(32, gru_hidden, batch_first=True)
        self.fc      = nn.Linear(gru_hidden, n_classes)

    def forward(self, x, h=None):
        x    = x.unsqueeze(1)
        feat = self.cnn(x).permute(0, 2, 1)
        feat = self.dropout(feat)
        out, h_new = self.gru(feat, h)
        return self.fc(out[:, -1, :]), h_new

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class TransitionDataset(torch.utils.data.Dataset):
    def __init__(self, npz_path, indices=None):
        d = np.load(npz_path)
        self.x_t  = d['x_t']
        self.x_t1 = d['x_t1']
        self.p_t  = d['p_t']
        self.p_t1 = d['p_t1']
        self.y_t  = d['y_t']
        self.w_t  = d['w_t']
        if indices is not None:
            for attr in ['x_t','x_t1','p_t','p_t1','y_t','w_t']:
                setattr(self, attr, getattr(self, attr)[indices])

    def __len__(self): return len(self.y_t)

    def __getitem__(self, idx):
        return {
            'x_t':  torch.FloatTensor(self.x_t[idx]),
            'x_t1': torch.FloatTensor(self.x_t1[idx]),
            'p_t':  torch.FloatTensor(self.p_t[idx]),
            'p_t1': torch.FloatTensor(self.p_t1[idx]),
            'y_t':  torch.LongTensor([self.y_t[idx]])[0],
            'w_t':  torch.FloatTensor([self.w_t[idx]])[0],
        }


def get_weight(p_t, scheme):
    if scheme == 'max_prob':
        return p_t.max(dim=-1).values
    elif scheme == 'uniform':
        return torch.ones(p_t.shape[0], device=p_t.device)
    elif scheme == 'entropy':
        H = -(p_t * (p_t + 1e-8).log()).sum(dim=-1)
        return 1.0 - (H / np.log(p_t.shape[-1]))
    elif scheme == 'margin':
        top2, _ = p_t.topk(2, dim=-1)
        return top2[:, 0] - top2[:, 1]
    else:
        raise ValueError(f"Unknown scheme: {scheme}")


def get_gamma_at_epoch(cfg, epoch, total_epochs):
    """
    Returns gamma for the current epoch — supports fixed or scheduled ramp-up.

    cfg['gamma_schedule']:
      'fixed'  (default) -> always cfg['gamma']
      'linear'           -> ramps gamma_start -> gamma_end linearly over warmup_frac
      'cosine'           -> same but with cosine easing (gentler start/end)

    Scheduled modes need: gamma_start, gamma_end, warmup_frac in cfg.
    """
    schedule = cfg.get('gamma_schedule', 'fixed')
    if schedule == 'fixed':
        return cfg['gamma']

    g_start = cfg.get('gamma_start', 0.0)
    g_end   = cfg.get('gamma_end', cfg.get('gamma', 0.8))
    warmup_frac = cfg.get('warmup_frac', 0.5)
    warmup_epochs = max(1, int(total_epochs * warmup_frac))

    if epoch >= warmup_epochs:
        return g_end

    progress = epoch / warmup_epochs
    if schedule == 'linear':
        return g_start + (g_end - g_start) * progress
    elif schedule == 'cosine':
        cos_progress = (1 - np.cos(progress * np.pi)) / 2
        return g_start + (g_end - g_start) * cos_progress
    else:
        raise ValueError(f"Unknown gamma_schedule: {schedule}")


def compute_loss(student, batch, device, cfg, return_components=False,
                 gamma_override=None):
    x_t  = batch['x_t'].to(device)
    x_t1 = batch['x_t1'].to(device)
    p_t  = batch['p_t'].to(device)
    p_t1 = batch['p_t1'].to(device)
    y_t  = batch['y_t'].to(device)

    alpha  = cfg['alpha']
    beta   = cfg['beta']
    gamma  = gamma_override if gamma_override is not None else cfg['gamma']
    tau    = cfg['tau']
    scheme = cfg.get('weight_scheme', 'max_prob')

    logits_t,  h = student(x_t)
    logits_t1, _ = student(x_t1, h.detach())

    L_ce  = F.cross_entropy(logits_t, y_t)

    log_q  = F.log_softmax(logits_t / tau, dim=-1)
    soft_p = F.softmax(p_t / tau, dim=-1)
    L_kd   = F.kl_div(log_q, soft_p, reduction='batchmean') * (tau ** 2)

    w_t       = get_weight(p_t, scheme)
    log_q1    = F.log_softmax(logits_t1 / tau, dim=-1)
    soft_p1   = F.softmax(p_t1 / tau, dim=-1)
    kl_per    = F.kl_div(log_q1, soft_p1, reduction='none').sum(dim=-1)
    L_trans   = (w_t * kl_per).mean() * (tau ** 2)

    L_total = alpha * L_ce + beta * L_kd + gamma * L_trans

    if return_components:
        return L_total, {
            'ce': L_ce.item(), 'kd': L_kd.item(),
            'trans': L_trans.item(), 'total': L_total.item()
        }
    return L_total


# -------------------------------------------------------------------
# Training
# -------------------------------------------------------------------
def train(model, train_dl, val_dl, device, cfg, epochs, save_path):
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=cfg.get('lr', 1e-3),
                                 weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs)

    history = {
        'train_loss': [], 'val_loss': [],
        'val_acc': [], 'val_kappa': [], 'val_f1': [],
        'loss_ce': [], 'loss_kd': [], 'loss_trans': [],
        'gamma_used': [],
    }
    best_kappa = -1.0

    for ep in range(1, epochs + 1):
        # Resolve gamma for this epoch (fixed or scheduled)
        gamma_ep = get_gamma_at_epoch(cfg, ep, epochs)

        model.train()
        ep_loss = {'ce':0,'kd':0,'trans':0,'total':0}
        n = 0
        for batch in tqdm(train_dl, desc=f"Ep {ep:3d}/{epochs} (γ={gamma_ep:.3f})",
                          leave=False):
            optimizer.zero_grad()
            loss, comps = compute_loss(model, batch, device, cfg,
                                       return_components=True,
                                       gamma_override=gamma_ep)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            for k in ep_loss: ep_loss[k] += comps.get(k, 0)
            n += 1
        scheduler.step()
        for k in ep_loss: ep_loss[k] /= n
        history['gamma_used'].append(gamma_ep)

        # Validate — use the SAME gamma_ep so val loss is comparable to train loss
        model.eval()
        preds_all, true_all = [], []
        val_loss_sum = 0
        with torch.no_grad():
            for batch in val_dl:
                loss, _ = compute_loss(model, batch, device, cfg,
                                       return_components=True,
                                       gamma_override=gamma_ep)
                val_loss_sum += loss.item()
                logits, _ = model(batch['x_t'].to(device))
                preds_all.extend(logits.argmax(-1).cpu().numpy())
                true_all.extend(batch['y_t'].numpy())

        preds_all = np.array(preds_all)
        true_all  = np.array(true_all)
        acc   = accuracy_score(true_all, preds_all)
        kappa = cohen_kappa_score(true_all, preds_all)
        f1    = f1_score(true_all, preds_all, average='macro', zero_division=0)
        vloss = val_loss_sum / len(val_dl)

        history['train_loss'].append(ep_loss['total'])
        history['val_loss'].append(vloss)
        history['val_acc'].append(acc)
        history['val_kappa'].append(kappa)
        history['val_f1'].append(f1)
        history['loss_ce'].append(ep_loss['ce'])
        history['loss_kd'].append(ep_loss['kd'])
        history['loss_trans'].append(ep_loss['trans'])

        print(f"  Ep {ep:3d} | loss {ep_loss['total']:.4f} "
              f"[ce={ep_loss['ce']:.3f} kd={ep_loss['kd']:.3f} "
              f"tr={ep_loss['trans']:.3f}] | "
              f"acc={acc:.4f} κ={kappa:.4f} F1={f1:.4f}")

        if kappa > best_kappa:
            best_kappa = kappa
            torch.save(model.state_dict(), save_path)

    model.load_state_dict(torch.load(save_path, map_location=device))
    return model, history


# -------------------------------------------------------------------
# Speed benchmark
# -------------------------------------------------------------------
def benchmark_speed(model, device, n_runs=500):
    model.eval()
    dummy = torch.randn(1, 3000).to(device)
    for _ in range(50):
        with torch.no_grad(): model(dummy)
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            model(dummy)
            times.append((time.perf_counter() - t0) * 1000)
    return {
        'mean_ms':   float(np.mean(times)),
        'std_ms':    float(np.std(times)),
        'median_ms': float(np.median(times)),
        'min_ms':    float(np.min(times)),
    }


# -------------------------------------------------------------------
# Serialization helper
# -------------------------------------------------------------------
def _serial(obj):
    if isinstance(obj, dict):   return {k: _serial(v) for k,v in obj.items()}
    if isinstance(obj, list):   return [_serial(i) for i in obj]
    if isinstance(obj, np.integer):  return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.ndarray):  return obj.tolist()
    return obj


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load best config from sweep
    sweep_path = "outputs/sweep_results.json"
    if not os.path.exists(sweep_path):
        print("Run 6_sweep.py first.")
        return

    with open(sweep_path) as f:
        sweep = json.load(f)

    best_cfg = sweep['best_config']
    print(f"\nBest config from sweep:")
    print(f"  γ      = {best_cfg['gamma']}")
    print(f"  τ      = {best_cfg['tau']}")
    print(f"  scheme = {best_cfg['weight_scheme']}")

    # Dataset — FIX: subject-level split (was leaky pair-index split)
    dataset_path = "outputs/transition_dataset.npz"
    from split_utils import subject_level_split, verify_no_leakage

    raw = np.load(dataset_path)
    subject_ids = raw['subject_id']

    train_idx, val_idx, train_subj, val_subj = subject_level_split(
        subject_ids, val_frac=0.2, seed=42)

    leak_frac = verify_no_leakage(raw['x_t'], raw['x_t1'], train_idx, val_idx)
    if leak_frac > 0.001:
        print("[ABORT] Leakage detected after subject split. Check split_utils.py.")
        return

    train_ds = TransitionDataset(dataset_path, indices=train_idx)
    val_ds   = TransitionDataset(dataset_path, indices=val_idx)
    train_dl = DataLoader(train_ds, batch_size=256, shuffle=True,  num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=256, shuffle=False, num_workers=0)

    os.makedirs("checkpoints", exist_ok=True)

    # -------------------------------------------------------------------
    # Train baseline (γ=0) and best sweep config side by side
    # So 5_evaluate.py gets a clean apples-to-apples comparison
    # -------------------------------------------------------------------
    # -------------------------------------------------------------------
    # Three configs:
    #   baseline_kd    : gamma=0, no transition loss
    #   proposed_tad   : fixed gamma from sweep (e.g. 0.8), uniform weighting
    #   proposed_dynamic: same end-gamma, but ramped in via cosine schedule
    #                     instead of applied at full strength from epoch 1
    # weight_scheme stays 'uniform' throughout per your decision -- only the
    # gamma schedule changes between proposed_tad and proposed_dynamic.
    # -------------------------------------------------------------------
    configs = {
        'baseline_kd': {
            **best_cfg,
            'gamma':          0.0,
            'gamma_schedule': 'fixed',
            'name':           'baseline_kd',
        },
        'proposed_tad': {
            **best_cfg,
            'gamma_schedule': 'fixed',
            'name': 'proposed_tad',
        },
        'proposed_dynamic': {
            **best_cfg,
            'gamma_schedule': 'cosine',
            'gamma_start':    0.0,
            'gamma_end':      best_cfg['gamma'],   # ramps up TO the sweep-best value
            'warmup_frac':    0.5,                  # reaches gamma_end by halfway
            'name':           'proposed_dynamic',
        },
    }

    results = {}

    for run_name, cfg in configs.items():
        print(f"\n{'='*60}")
        print(f"Training: {run_name}")
        gamma_desc = (f"γ={cfg['gamma']}" if cfg.get('gamma_schedule','fixed') == 'fixed'
                     else f"γ: {cfg.get('gamma_start',0)}→{cfg.get('gamma_end')} "
                          f"({cfg['gamma_schedule']}, warmup={cfg.get('warmup_frac')})")
        print(f"  {gamma_desc}  τ={cfg['tau']}  "
              f"scheme={cfg.get('weight_scheme','max_prob')}")
        print(f"{'='*60}")

        model = SleepStudentNet().to(device)
        save_path = f"checkpoints/student_{run_name}_sweep_best.pt"

        model, history = train(
            model, train_dl, val_dl, device, cfg,
            epochs=60, save_path=save_path)

        # Final evaluation
        model.eval()
        all_preds, all_true, all_probs = [], [], []

        with torch.no_grad():
            for batch in val_dl:
                logits, _ = model(batch['x_t'].to(device))
                probs = F.softmax(logits, dim=-1).cpu().numpy()
                all_preds.extend(logits.argmax(-1).cpu().numpy())
                all_true.extend(batch['y_t'].numpy())
                all_probs.extend(probs)

        all_preds = np.array(all_preds)
        all_true  = np.array(all_true)
        all_probs = np.array(all_probs)

        stage_f1 = f1_score(all_true, all_preds, average=None,
                             labels=[0,1,2,3,4], zero_division=0)
        speed = benchmark_speed(model, device)

        results[run_name] = {
            'config':      cfg,
            'history':     history,
            'params':      model.count_params(),
            'accuracy':    float(accuracy_score(all_true, all_preds)),
            'kappa':       float(cohen_kappa_score(all_true, all_preds)),
            'macro_f1':    float(f1_score(all_true, all_preds,
                                          average='macro', zero_division=0)),
            'stage_f1':    stage_f1.tolist(),
            'speed_ms':    speed,
            'predictions': all_preds.tolist(),
            'true_labels': all_true.tolist(),
            'probs':       all_probs.tolist(),
        }

        r = results[run_name]
        print(f"\n[{run_name}]")
        print(f"  Accuracy : {r['accuracy']:.4f}")
        print(f"  Kappa    : {r['kappa']:.4f}")
        print(f"  Macro F1 : {r['macro_f1']:.4f}")
        print(f"  Per-stage: " +
              " | ".join([f"{s}={v:.3f}"
                          for s,v in zip(["W","N1","N2","N3","REM"],
                                         r['stage_f1'])]))
        print(f"  Speed    : {speed['mean_ms']:.3f} ms/epoch")

    # Save — overwrite training_results.json so 5_evaluate.py picks it up
    with open("outputs/training_results.json", "w") as f:
        json.dump(_serial(results), f, indent=2)

    print("\nSaved: outputs/training_results.json  (overwritten with sweep-best results)")
    print("\nNow run:  python 5_evaluate.py")
    print("This will regenerate ALL plots with the tuned model.")

    # Print delta vs baseline for both fixed and dynamic gamma
    base_k  = results['baseline_kd']['kappa']
    fixed_k = results['proposed_tad']['kappa']
    dyn_k   = results['proposed_dynamic']['kappa']
    base_n1  = results['baseline_kd']['stage_f1'][1]
    fixed_n1 = results['proposed_tad']['stage_f1'][1]
    dyn_n1   = results['proposed_dynamic']['stage_f1'][1]
    base_rem  = results['baseline_kd']['stage_f1'][4]
    fixed_rem = results['proposed_tad']['stage_f1'][4]
    dyn_rem   = results['proposed_dynamic']['stage_f1'][4]

    print("\n" + "="*55)
    print("THREE-WAY COMPARISON")
    print("="*55)
    print(f"{'Config':<20} {'Kappa':>8} {'ΔKappa':>9} "
          f"{'N1 F1':>7} {'REM F1':>8}")
    print(f"{'Baseline (γ=0)':<20} {base_k:>8.4f} {'—':>9} "
          f"{base_n1:>7.3f} {base_rem:>8.3f}")
    print(f"{'Fixed γ':<20} {fixed_k:>8.4f} {fixed_k-base_k:>+9.4f} "
          f"{fixed_n1:>7.3f} {fixed_rem:>8.3f}")
    print(f"{'Dynamic γ (cosine)':<20} {dyn_k:>8.4f} {dyn_k-base_k:>+9.4f} "
          f"{dyn_n1:>7.3f} {dyn_rem:>8.3f}")

    


if __name__ == "__main__":
    main()