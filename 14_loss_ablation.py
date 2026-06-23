import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import cohen_kappa_score, f1_score, accuracy_score
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from split_utils import subject_level_split, verify_no_leakage

STAGE_NAMES = ["Wake", "N1", "N2", "N3", "REM"]
N_CLASSES   = 5

DARK_BG   = "#0f1117"
PANEL_BG  = "#1a1d27"
GRID_COL  = "#2a2d3a"
TEXT_COL  = "#e8eaf0"
TEXT_DIM  = "#8890a8"

# Consistent colors for each config across all plots
CONFIG_COLORS = {
    'ce_only':  "#5b8dee",   # blue
    'full_tad': "#e8643a",   # coral
    'no_kd':    "#82b366",   # green
    'teacher':  "#f5c842",   # gold
}
CONFIG_LABELS = {
    'ce_only':  "CE only  (β=0, γ=0)",
    'full_tad': "Full TAD  (CE + KD + trans)",
    'no_kd':    "No KD  (β=0, γ=0.2)",
    'teacher':  "Teacher (TinySleepNet)",
}

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
        if indices is not None:
            for attr in ['x_t','x_t1','p_t','p_t1','y_t']:
                setattr(self, attr, getattr(self, attr)[indices])

    def __len__(self): return len(self.y_t)

    def __getitem__(self, idx):
        return {
            'x_t':  torch.FloatTensor(self.x_t[idx]),
            'x_t1': torch.FloatTensor(self.x_t1[idx]),
            'p_t':  torch.FloatTensor(self.p_t[idx]),
            'p_t1': torch.FloatTensor(self.p_t1[idx]),
            'y_t':  torch.LongTensor([self.y_t[idx]])[0],
        }

# -------------------------------------------------------------------
# Loss
# -------------------------------------------------------------------
def compute_loss(student, batch, device, alpha, beta, gamma, tau=2.0,
                 return_components=False):
    x_t  = batch['x_t'].to(device)
    x_t1 = batch['x_t1'].to(device)
    p_t  = batch['p_t'].to(device)
    p_t1 = batch['p_t1'].to(device)
    y_t  = batch['y_t'].to(device)

    logits_t,  h = student(x_t)
    logits_t1, _ = student(x_t1, h.detach())

    L_ce = F.cross_entropy(logits_t, y_t)

    if beta > 0:
        log_q  = F.log_softmax(logits_t / tau, dim=-1)
        soft_p = F.softmax(p_t / tau, dim=-1)
        L_kd   = F.kl_div(log_q, soft_p, reduction='batchmean') * (tau ** 2)
    else:
        L_kd = torch.tensor(0.0, device=device)

    if gamma > 0:
        log_q1  = F.log_softmax(logits_t1 / tau, dim=-1)
        soft_p1 = F.softmax(p_t1 / tau, dim=-1)
        kl_per  = F.kl_div(log_q1, soft_p1, reduction='none').sum(dim=-1)
        L_trans = kl_per.mean() * (tau ** 2)
    else:
        L_trans = torch.tensor(0.0, device=device)

    L_total = alpha * L_ce + beta * L_kd + gamma * L_trans

    if return_components:
        return L_total, {
            'ce':    L_ce.item(),
            'kd':    L_kd.item(),
            'trans': L_trans.item(),
            'total': L_total.item(),
        }
    return L_total

def train_run(cfg, train_dl, val_dl, device, save_path, epochs=40):
    model = SleepStudentNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    alpha = cfg['alpha']
    beta  = cfg['beta']
    gamma = cfg['gamma']

    history = {
        'train_loss': [], 'val_acc': [], 'val_kappa': [], 'val_f1': [],
        'loss_ce': [], 'loss_kd': [], 'loss_trans': [],
        'stage_f1_per_epoch': [],
    }
    best_kappa = -1.0
    best_metrics = {}

    print(f"\n{'='*60}")
    print(f"Training: {cfg['name']}")
    print(f"  alpha={alpha}  beta={beta}  gamma={gamma}")
    print(f"  Params: {model.count_params():,}")
    print(f"{'='*60}")

    for ep in range(1, epochs + 1):
        model.train()
        ep_loss = {'ce':0,'kd':0,'trans':0,'total':0}
        n = 0
        for batch in tqdm(train_dl, desc=f"Ep {ep:3d}/{epochs}", leave=False):
            optimizer.zero_grad()
            loss, comps = compute_loss(model, batch, device,
                                       alpha, beta, gamma,
                                       return_components=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            for k in ep_loss: ep_loss[k] += comps.get(k, 0)
            n += 1
        scheduler.step()
        for k in ep_loss: ep_loss[k] /= n

        model.eval()
        preds_all, true_all = [], []
        with torch.no_grad():
            for batch in val_dl:
                logits, _ = model(batch['x_t'].to(device))
                preds_all.extend(logits.argmax(-1).cpu().numpy())
                true_all.extend(batch['y_t'].numpy())

        preds_all = np.array(preds_all); true_all = np.array(true_all)
        acc      = accuracy_score(true_all, preds_all)
        kappa    = cohen_kappa_score(true_all, preds_all)
        f1       = f1_score(true_all, preds_all, average='macro', zero_division=0)
        stage_f1 = f1_score(true_all, preds_all, average=None,
                            labels=[0,1,2,3,4], zero_division=0)

        history['train_loss'].append(ep_loss['total'])
        history['val_acc'].append(acc)
        history['val_kappa'].append(kappa)
        history['val_f1'].append(f1)
        history['loss_ce'].append(ep_loss['ce'])
        history['loss_kd'].append(ep_loss['kd'])
        history['loss_trans'].append(ep_loss['trans'])
        history['stage_f1_per_epoch'].append(stage_f1.tolist())

        print(f"  Ep {ep:3d} | "
              f"ce={ep_loss['ce']:.4f} kd={ep_loss['kd']:.4f} "
              f"tr={ep_loss['trans']:.4f} | "
              f"acc={acc:.4f} κ={kappa:.4f} F1={f1:.4f} "
              f"N1={stage_f1[1]:.3f}")

        if kappa > best_kappa:
            best_kappa = kappa
            best_metrics = {
                'accuracy': float(acc), 'kappa': float(kappa),
                'macro_f1': float(f1), 'stage_f1': stage_f1.tolist(),
                'epoch': ep,
            }
            torch.save(model.state_dict(), save_path)

    model.load_state_dict(torch.load(save_path, map_location=device))
    return model, history, best_metrics

def set_ax(ax, title, xlabel=None, ylabel=None):
    ax.set_facecolor(PANEL_BG)
    ax.set_title(title, color=TEXT_COL, fontsize=12, fontweight='bold')
    ax.tick_params(colors=TEXT_DIM)
    ax.grid(color=GRID_COL, linewidth=0.6)
    for spine in ax.spines.values(): spine.set_edgecolor(GRID_COL)
    if xlabel: ax.set_xlabel(xlabel, color=TEXT_DIM, fontsize=10)
    if ylabel: ax.set_ylabel(ylabel, color=TEXT_DIM, fontsize=10)


def plot_loss_curves(results, out_path):
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.patch.set_facecolor(DARK_BG)
    fig.suptitle("Loss Term Ablation — Training Curves",
                 fontsize=14, fontweight='bold', color=TEXT_COL)

    epochs_range = None

    ax = axes[0]
    set_ax(ax, "Total Training Loss", xlabel="Epoch", ylabel="Loss")
    for name in ['ce_only', 'full_tad', 'no_kd']:
        if name not in results: continue
        y = results[name]['history']['train_loss']
        if epochs_range is None: epochs_range = range(1, len(y)+1)
        ax.plot(epochs_range, y, color=CONFIG_COLORS[name],
                linewidth=2, label=CONFIG_LABELS[name])
    ax.legend(fontsize=8, facecolor=PANEL_BG, labelcolor=TEXT_COL,
             edgecolor=GRID_COL)
    ax2 = axes[1]
    set_ax(ax2, "CE Loss Component\n(comparable across all configs)",
           xlabel="Epoch", ylabel="L_CE (unweighted)")
    for name in ['ce_only', 'full_tad', 'no_kd']:
        if name not in results: continue
        y = results[name]['history']['loss_ce']
        ax2.plot(range(1, len(y)+1), y, color=CONFIG_COLORS[name],
                 linewidth=2, label=CONFIG_LABELS[name])
    ax2.legend(fontsize=8, facecolor=PANEL_BG, labelcolor=TEXT_COL,
              edgecolor=GRID_COL)

    ax3 = axes[2]
    set_ax(ax3, "KD & Transition Loss Components\n(for configs that use them)",
           xlabel="Epoch", ylabel="Component Loss")
    if 'full_tad' in results:
        y_kd = results['full_tad']['history']['loss_kd']
        y_tr = results['full_tad']['history']['loss_trans']
        ax3.plot(range(1, len(y_kd)+1), y_kd,
                 color=CONFIG_COLORS['full_tad'], linewidth=2,
                 linestyle='-',  label="Full TAD — L_KD")
        ax3.plot(range(1, len(y_tr)+1), y_tr,
                 color=CONFIG_COLORS['full_tad'], linewidth=2,
                 linestyle='--', label="Full TAD — L_trans")
    if 'no_kd' in results:
        y_tr2 = results['no_kd']['history']['loss_trans']
        ax3.plot(range(1, len(y_tr2)+1), y_tr2,
                 color=CONFIG_COLORS['no_kd'], linewidth=2,
                 linestyle='--', label="No KD — L_trans")
    ax3.legend(fontsize=8, facecolor=PANEL_BG, labelcolor=TEXT_COL,
              edgecolor=GRID_COL)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
    plt.close()
    print(f"Saved: {out_path}")


def plot_stage_f1(results, teacher_stage_f1, out_path):
    fig, ax = plt.subplots(figsize=(13, 6))
    fig.patch.set_facecolor(DARK_BG)
    set_ax(ax, "Per-Stage F1 Comparison — Loss Ablation\n"
               "(dashed lines = teacher reference per stage)",
           xlabel="Sleep Stage", ylabel="F1 Score")

    configs = ['ce_only', 'full_tad', 'no_kd']
    n_configs = len(configs)
    x = np.arange(N_CLASSES)
    width = 0.22

    for i, name in enumerate(configs):
        if name not in results: continue
        sf1 = results[name]['best']['stage_f1']
        offset = (i - n_configs/2 + 0.5) * width
        bars = ax.bar(x + offset, sf1, width,
                      color=CONFIG_COLORS[name], alpha=0.85,
                      label=CONFIG_LABELS[name], zorder=3)
        for bar, val in zip(bars, sf1):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.008,
                    f"{val:.3f}", ha='center', va='bottom',
                    fontsize=7, color=TEXT_DIM)

    # Teacher reference lines per stage
    if teacher_stage_f1 is not None:
        for j, (stage, t_f1) in enumerate(zip(STAGE_NAMES, teacher_stage_f1)):
            ax.hlines(t_f1, j - 0.4, j + 0.4,
                      colors=CONFIG_COLORS['teacher'],
                      linewidth=2, linestyles='--', zorder=4)
        ax.plot([], [], color=CONFIG_COLORS['teacher'], linewidth=2,
                linestyle='--', label=CONFIG_LABELS['teacher'])

    ax.set_xticks(x)
    ax.set_xticklabels(STAGE_NAMES, color=TEXT_COL, fontsize=12)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=9, facecolor=PANEL_BG, labelcolor=TEXT_COL,
             edgecolor=GRID_COL, loc='lower right')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
    plt.close()
    print(f"Saved: {out_path}")


def plot_accuracy_kappa(results, teacher_acc, teacher_kappa, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor(DARK_BG)
    fig.suptitle("Accuracy & Kappa — Loss Term Ablation",
                 fontsize=13, fontweight='bold', color=TEXT_COL)

    configs = ['ce_only', 'full_tad', 'no_kd']
    labels  = [CONFIG_LABELS[c].replace("  ", "\n") for c in configs]
    colors  = [CONFIG_COLORS[c] for c in configs]

    for ax, metric, title, teacher_val in zip(
        axes,
        ['accuracy', 'kappa'],
        ['Accuracy', 'Cohen κ'],
        [teacher_acc, teacher_kappa]
    ):
        set_ax(ax, title, ylabel=title)
        vals = [results[c]['best'].get(metric, 0) if c in results else 0
                for c in configs]
        bars = ax.bar(labels, vals, color=colors, alpha=0.85, zorder=3, width=0.5)

        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.002,
                    f"{val:.4f}", ha='center', va='bottom',
                    fontsize=11, color=TEXT_COL, fontweight='bold')

        # Teacher reference line
        if teacher_val is not None:
            ax.axhline(teacher_val, color=CONFIG_COLORS['teacher'],
                       linewidth=2, linestyle='--',
                       label=f"Teacher ({teacher_val:.4f})")
            ax.legend(fontsize=9, facecolor=PANEL_BG, labelcolor=TEXT_COL,
                     edgecolor=GRID_COL)

        # Set y range with a bit of headroom
        all_vals = vals + ([teacher_val] if teacher_val else [])
        ax.set_ylim(min(all_vals) - 0.02, max(all_vals) + 0.03)
        ax.tick_params(axis='x', colors=TEXT_COL, labelsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
    plt.close()
    print(f"Saved: {out_path}")

def _serial(obj):
    if isinstance(obj, dict): return {k:_serial(v) for k,v in obj.items()}
    if isinstance(obj, list): return [_serial(i) for i in obj]
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    return obj


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset_path = "outputs/transition_dataset.npz"
    if not os.path.exists(dataset_path):
        print("Run steps 1-3 first.")
        return

    raw = np.load(dataset_path)
    subject_ids = raw['subject_id']
    train_idx, val_idx, train_subj, val_subj = subject_level_split(
        subject_ids, val_frac=0.2, seed=42)
    leak = verify_no_leakage(raw['x_t'], raw['x_t1'], train_idx, val_idx)
    if leak > 0.001:
        print("Leakage detected."); return

    train_ds = TransitionDataset(dataset_path, indices=train_idx)
    val_ds   = TransitionDataset(dataset_path, indices=val_idx)
    train_dl = DataLoader(train_ds, batch_size=256, shuffle=True,  num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=256, shuffle=False, num_workers=0)
    print(f"Dataset: {len(train_ds):,} train | {len(val_ds):,} val pairs")

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("plots", exist_ok=True)

    configs = {
        'ce_only': {
            'name': 'ce_only',
            'alpha': 1.0, 'beta': 0.0, 'gamma': 0.0,
        },
        'full_tad': {
            'name': 'full_tad',
            'alpha': 0.3, 'beta': 0.5, 'gamma': 0.2,
        },
        'no_kd': {
            'name': 'no_kd',
            'alpha': 0.3, 'beta': 0.0, 'gamma': 0.2,
        },
    }

    results = {}
    for name, cfg in configs.items():
        save_path = f"checkpoints/student_ablation_{name}.pt"
        model, history, best = train_run(
            cfg, train_dl, val_dl, device, save_path, epochs=40)
        results[name] = {
            'config': cfg, 'history': history, 'best': best,
            'params': model.count_params(),
        }

    with open("outputs/ablation_results.json", "w") as f:
        json.dump(_serial(results), f, indent=2)

    teacher_stage_f1 = None
    teacher_acc      = None
    teacher_kappa    = None

    teacher_metrics_path = "outputs/teacher_val_metrics.json"
    if os.path.exists(teacher_metrics_path):
        with open(teacher_metrics_path) as f:
            tm = json.load(f)
        teacher_stage_f1 = tm['stage_f1']
        teacher_acc      = tm['accuracy']
        teacher_kappa    = tm['kappa']
        print(f"\nTeacher reference: acc={teacher_acc:.4f} "
              f"κ={teacher_kappa:.4f} F1={tm['macro_f1']:.4f}")
    else:
        print("\nNo teacher_val_metrics.json found.")
    plot_loss_curves(results, "plots/ablation_loss_curves.png")
    plot_stage_f1(results, teacher_stage_f1, "plots/ablation_stage_f1.png")
    plot_accuracy_kappa(results, teacher_acc, teacher_kappa,
                        "plots/ablation_accuracy_kappa.png")

    print(f"{'Config':<12}{'alpha':>7}{'beta':>7}{'gamma':>7}"
          f"{'Acc':>8}{'Kappa':>8}{'MacroF1':>9}"
          f"{'N1 F1':>8}{'REM F1':>8}{'Epoch':>7}")
    for name, cfg in configs.items():
        b = results[name]['best']
        print(f"{name:<12}{cfg['alpha']:>7.1f}{cfg['beta']:>7.1f}"
              f"{cfg['gamma']:>7.1f}"
              f"{b['accuracy']:>8.4f}{b['kappa']:>8.4f}"
              f"{b['macro_f1']:>9.4f}{b['stage_f1'][1]:>8.3f}"
              f"{b['stage_f1'][4]:>8.3f}{b['epoch']:>7}")

    print(f"\nSaved: outputs/ablation_results.json")
    print("Plots:")
    print("  plots/ablation_loss_curves.png")
    print("  plots/ablation_stage_f1.png")
    print("  plots/ablation_accuracy_kappa.png")

if __name__ == "__main__":
    main()