"""
Step 3 : Build transition dataset from teacher soft labels.
"""

import os
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import matplotlib.patheffects as pe
import math

STAGE_NAMES  = ["Wake", "N1", "N2", "N3", "REM"]
N_CLASSES    = 5

def build_subject_transitions(X, y, soft_labels, subject_id):
    """
    X            : (N, 3000) raw EEG epochs
    y            : (N,)      hard ground-truth labels
    soft_labels  : (N, 5)    teacher softmax outputs p_t^T
    subject_id   : str       identifier for this recording

    Returns list of dicts, one per consecutive pair, each tagged with subject_id.
    """
    N = len(X)
    pairs = []
    for t in range(N - 1):
        s_t = int(np.argmax(soft_labels[t]))
        w_t = float(np.max(soft_labels[t]))

        pairs.append({
            'x_t':        X[t],
            'x_t1':       X[t + 1],
            'p_t':        soft_labels[t],
            'p_t1':       soft_labels[t + 1],
            'y_t':        int(y[t]),
            'y_t1':       int(y[t + 1]),
            's_t':        s_t,
            'w_t':        w_t,
            'subject_id': subject_id,     
        })
    return pairs

def compute_transition_matrix(all_pairs):
    M_sum   = np.zeros((N_CLASSES, N_CLASSES), dtype=np.float64)
    M_count = np.zeros(N_CLASSES, dtype=np.float64)
    for pair in all_pairs:
        s_i = pair['s_t']
        M_sum[s_i]   += pair['p_t1']
        M_count[s_i] += 1
    M = np.zeros((N_CLASSES, N_CLASSES), dtype=np.float32)
    for i in range(N_CLASSES):
        M[i] = M_sum[i] / M_count[i] if M_count[i] > 0 else np.eye(N_CLASSES)[i]
    return M

def compute_transition_matrix_with_std(all_pairs):
    """
    Returns:
        M_mean : (5, 5) float32 -- same semantics as compute_transition_matrix
        M_std  : (5, 5) float32 -- per-cell standard deviation across epochs
        M_n    : (5,)   int     -- number of supporting epochs per source state
                                    (useful to flag unreliable/sparse cells)
    """
    n     = np.zeros(N_CLASSES, dtype=np.int64)
    mean  = np.zeros((N_CLASSES, N_CLASSES), dtype=np.float64)
    M2    = np.zeros((N_CLASSES, N_CLASSES), dtype=np.float64)  # sum of squared diffs from mean

    for pair in all_pairs:
        s_i = pair['s_t']
        x   = pair['p_t1'].astype(np.float64)   # (5,) vector for this occurrence

        n[s_i] += 1
        delta      = x - mean[s_i]
        mean[s_i] += delta / n[s_i]
        delta2     = x - mean[s_i]
        M2[s_i]   += delta * delta2

    M_mean = np.zeros((N_CLASSES, N_CLASSES), dtype=np.float32)
    M_std  = np.zeros((N_CLASSES, N_CLASSES), dtype=np.float32)

    for i in range(N_CLASSES):
        if n[i] > 1:
            M_mean[i] = mean[i].astype(np.float32)
            variance  = M2[i] / (n[i] - 1)   # sample variance (Bessel's correction)
            M_std[i]  = np.sqrt(np.maximum(variance, 0)).astype(np.float32)
        elif n[i] == 1:
            M_mean[i] = mean[i].astype(np.float32)
            M_std[i]  = 0.0   # single sample, no variance estimate possible
        else:
            M_mean[i] = np.eye(N_CLASSES)[i]   # fallback: never observed
            M_std[i]  = 0.0

    return M_mean, M_std, n

def plot_transition_graph(M, out_path="plots/transition_graph.png"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor('#0f1117')

    COLORS = {"Wake": "#6c8ebf", "N1": "#82b366", "N2": "#d6a920",
              "N3": "#b03f30", "REM": "#9b59b6"}

    ax = axes[0]
    ax.set_facecolor('#0f1117')
    ax.set_xlim(-1.6, 1.6); ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal'); ax.axis('off')
    ax.set_title("Teacher Transition Policy Graph", color='white',
                 fontsize=14, fontweight='bold', pad=12)

    angles = [90, 162, 234, 306, 18]
    positions = {}
    for i, name in enumerate(STAGE_NAMES):
        rad = math.radians(angles[i])
        positions[name] = (math.cos(rad), math.sin(rad))

    THRESH = 0.04
    for i, src in enumerate(STAGE_NAMES):
        for j, dst in enumerate(STAGE_NAMES):
            prob = float(M[i, j])
            if prob < THRESH or src == dst:
                continue
            x0, y0 = positions[src]; x1, y1 = positions[dst]
            arr = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                  connectionstyle="arc3,rad=0.2",
                                  color=COLORS[src], alpha=max(0.15, prob),
                                  linewidth=prob * 8, mutation_scale=15, zorder=1)
            ax.add_patch(arr)
            mx = (x0+x1)/2 + 0.08*(y1-y0); my = (y0+y1)/2 + 0.08*(x0-x1)
            if prob > 0.08:
                ax.text(mx, my, f"{prob:.2f}", fontsize=7, color='white',
                        ha='center', va='center', alpha=0.85,
                        path_effects=[pe.withStroke(linewidth=2, foreground='#0f1117')])

    for i, name in enumerate(STAGE_NAMES):
        prob = float(M[i, i])
        if prob < THRESH: continue
        x0, y0 = positions[name]
        theta = math.radians(angles[i])
        dx, dy = math.cos(theta)*0.25, math.sin(theta)*0.25
        arc = matplotlib.patches.Arc((x0+dx*0.6, y0+dy*0.6), width=0.35, height=0.35,
                                     angle=math.degrees(theta)+90, theta1=0, theta2=270,
                                     color=COLORS[name], linewidth=prob*6,
                                     alpha=max(0.2, prob))
        ax.add_patch(arc)
        ax.text(x0+dx*0.9, y0+dy*0.9, f"{prob:.2f}", fontsize=7, color='white',
                ha='center', va='center',
                path_effects=[pe.withStroke(linewidth=2, foreground='#0f1117')])

    for name in STAGE_NAMES:
        x, y = positions[name]
        ax.add_patch(plt.Circle((x, y), 0.18, color=COLORS[name], zorder=3))
        ax.text(x, y, name, ha='center', va='center', fontsize=10,
                color='white', fontweight='bold', zorder=4)

    ax2 = axes[1]
    ax2.set_facecolor('#0f1117')
    im = ax2.imshow(M, cmap='YlOrRd', vmin=0, vmax=1, aspect='auto')
    ax2.set_xticks(range(N_CLASSES)); ax2.set_yticks(range(N_CLASSES))
    ax2.set_xticklabels(STAGE_NAMES, color='white', fontsize=11)
    ax2.set_yticklabels(STAGE_NAMES, color='white', fontsize=11)
    ax2.set_xlabel("Next stage  s_{t+1}", color='white', fontsize=12)
    ax2.set_ylabel("Current stage  s_t",  color='white', fontsize=12)
    ax2.set_title("Transition Matrix M (Eq. 6)", color='white',
                  fontsize=14, fontweight='bold')
    ax2.tick_params(colors='white')
    for spine in ax2.spines.values(): spine.set_edgecolor('#444')

    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            val = M[i, j]
            ax2.text(j, i, f"{val:.3f}", ha='center', va='center', fontsize=9,
                     color='black' if val > 0.5 else 'white', fontweight='bold')

    cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color='white')
    cbar.outline.set_edgecolor('white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='#0f1117')
    plt.close()
    print(f"Transition graph saved: {out_path}")


def plot_std_matrix(M_mean, M_std, M_n, out_path="plots/transition_std_matrix.png"):
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.patch.set_facecolor('#0f1117')

    panels = [
        (M_mean, "Mean  M[i,j]\n(point estimate, as before)", 'YlOrRd', 0, 1),
        (M_std,  "Std Dev  σ[i,j]\n(teacher consistency per transition)", 'BuPu', 0, None),
    ]

    for ax, (mat, title, cmap, vmin, vmax) in zip(axes[:2], panels):
        im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
        ax.set_xticks(range(N_CLASSES)); ax.set_yticks(range(N_CLASSES))
        ax.set_xticklabels(STAGE_NAMES, color='white', fontsize=10)
        ax.set_yticklabels(STAGE_NAMES, color='white', fontsize=10)
        ax.set_xlabel("Next stage  s_{t+1}", color='white', fontsize=11)
        ax.set_ylabel("Current stage  s_t",  color='white', fontsize=11)
        ax.set_title(title, color='white', fontsize=12, fontweight='bold')
        ax.tick_params(colors='white')
        for spine in ax.spines.values(): spine.set_edgecolor('#444')
        for i in range(N_CLASSES):
            for j in range(N_CLASSES):
                val = mat[i, j]
                thresh = 0.5 if vmax == 1 else mat.max() * 0.5
                ax.text(j, i, f"{val:.3f}", ha='center', va='center', fontsize=8.5,
                        color='black' if val > thresh else 'white', fontweight='bold')
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

    ax3 = axes[2]
    with np.errstate(divide='ignore', invalid='ignore'):
        cv = np.where(M_mean > 1e-6, M_std / (M_mean + 1e-8), 0)
    im3 = ax3.imshow(cv, cmap='RdYlGn_r', vmin=0, vmax=np.percentile(cv, 95), aspect='auto')
    ax3.set_xticks(range(N_CLASSES)); ax3.set_yticks(range(N_CLASSES))
    ax3.set_xticklabels(STAGE_NAMES, color='white', fontsize=10)
    ax3.set_yticklabels(STAGE_NAMES, color='white', fontsize=10)
    ax3.set_xlabel("Next stage  s_{t+1}", color='white', fontsize=11)
    ax3.set_ylabel("Current stage  s_t",  color='white', fontsize=11)
    ax3.set_title("Coefficient of Variation  σ/μ\n(relative uncertainty -- higher = less trustworthy)",
                  color='white', fontsize=12, fontweight='bold')
    ax3.tick_params(colors='white')
    for spine in ax3.spines.values(): spine.set_edgecolor('#444')
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            n_val = M_n[i]
            label = f"{cv[i,j]:.2f}\n(n={n_val})" if j == 0 else f"{cv[i,j]:.2f}"
            ax3.text(j, i, label, ha='center', va='center', fontsize=7.5,
                     color='white', fontweight='bold')
    cbar3 = plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    cbar3.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar3.ax.axes, 'yticklabels'), color='white')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='#0f1117')
    plt.close()
    print(f"Std/uncertainty matrix saved: {out_path}")


def plot_hypnogram(y_true, y_teacher, subj_id, out_path):
    fig, axes = plt.subplots(2, 1, figsize=(14, 4), sharex=True)
    fig.patch.set_facecolor('#0f1117')
    stage_colors = ["#6c8ebf", "#82b366", "#d6a920", "#ae4132", "#9b59b6"]
    time_hrs = np.arange(len(y_true)) * 30 / 3600

    for ax, y, title in zip(axes, [y_true, y_teacher],
                             ["Ground Truth", "Teacher (TinySleepNet)"]):
        ax.set_facecolor('#0f1117')
        for t, stage in enumerate(y):
            ax.axvspan(time_hrs[t], time_hrs[t]+30/3600,
                       ymin=(4-stage)/5, ymax=(5-stage)/5,
                       color=stage_colors[stage], alpha=0.85)
        ax.set_yticks([0.5/5 + i/5 for i in range(5)])
        ax.set_yticklabels(STAGE_NAMES[::-1], color='white', fontsize=9)
        ax.set_title(title, color='white', fontsize=11)
        ax.tick_params(colors='white')
        for spine in ax.spines.values(): spine.set_edgecolor('#333')

    axes[1].set_xlabel("Time (hours)", color='white')
    plt.suptitle(f"Hypnogram — Subject {subj_id}", color='white', fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight', facecolor='#0f1117')
    plt.close()

def main():
    data_dir  = "data/processed"
    label_dir = "outputs/soft_labels"
    out_dir   = "outputs"
    plot_dir  = "plots"
    os.makedirs(out_dir,  exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    soft_files = sorted(glob.glob(os.path.join(label_dir, "*_soft.npy")))

    subj_ids = [os.path.basename(f).replace("_soft.npy", "") for f in soft_files]
    print(f"Building transition dataset for {len(subj_ids)} subjects.")
    print(f"Subjects: {subj_ids}")

    all_pairs = []

    for idx, subj_id in enumerate(subj_ids):
        npz_path  = os.path.join(data_dir,  f"{subj_id}.npz")
        soft_path = os.path.join(label_dir, f"{subj_id}_soft.npy")
        if not os.path.exists(npz_path) or not os.path.exists(soft_path):
            continue

        d = np.load(npz_path)
        X, y = d['x'], d['y']
        soft = np.load(soft_path)

        min_len = min(len(X), len(soft))
        X, y, soft = X[:min_len], y[:min_len], soft[:min_len]

        pairs = build_subject_transitions(X, y, soft, subject_id=subj_id)
        all_pairs.extend(pairs)

        if idx < 3:
            teacher_preds = soft.argmax(axis=1)
            plot_hypnogram(y, teacher_preds, subj_id,
                           os.path.join(plot_dir, f"hypnogram_{subj_id}.png"))

    # --- Compute mean AND std transition matrices (replaces mean-only M) ---
    M_mean, M_std, M_n = compute_transition_matrix_with_std(all_pairs)

    # Flag cells with too few supporting epochs to trust the std estimate
    MIN_RELIABLE_N = 30
    unreliable_mask = M_n < MIN_RELIABLE_N
    if unreliable_mask.any():
        print(f"\n[NOTE] Source states with < {MIN_RELIABLE_N} supporting epochs "
              f"(std estimate unreliable for these rows):")
        for i in np.where(unreliable_mask)[0]:
            print(f"  {STAGE_NAMES[i]}: only {M_n[i]} epochs")

    np.save(os.path.join(out_dir, "transition_matrix.npy"), M_mean)        # backward compatible
    np.save(os.path.join(out_dir, "transition_matrix_mean.npy"), M_mean)   # explicit name
    np.save(os.path.join(out_dir, "transition_matrix_std.npy"),  M_std)
    np.save(os.path.join(out_dir, "transition_matrix_n.npy"),    M_n)

    N_pairs = len(all_pairs)
    dataset = {
        'x_t':        np.array([p['x_t']        for p in all_pairs], dtype=np.float32),
        'x_t1':       np.array([p['x_t1']       for p in all_pairs], dtype=np.float32),
        'p_t':        np.array([p['p_t']        for p in all_pairs], dtype=np.float32),
        'p_t1':       np.array([p['p_t1']       for p in all_pairs], dtype=np.float32),
        'y_t':        np.array([p['y_t']        for p in all_pairs], dtype=np.int64),
        'y_t1':       np.array([p['y_t1']       for p in all_pairs], dtype=np.int64),
        's_t':        np.array([p['s_t']        for p in all_pairs], dtype=np.int64),
        'w_t':        np.array([p['w_t']        for p in all_pairs], dtype=np.float32),
        'subject_id': np.array([p['subject_id'] for p in all_pairs], dtype='<U10'),
    }
    np.savez_compressed(os.path.join(out_dir, "transition_dataset.npz"), **dataset)

    plot_transition_graph(M_mean, out_path=os.path.join(plot_dir, "transition_graph.png"))
    plot_std_matrix(M_mean, M_std, M_n,
                    out_path=os.path.join(plot_dir, "transition_std_matrix.png"))

    print("\nPopulation Transition Matrix M (mean):")
    print(f"{'':>6}", end="")
    for n in STAGE_NAMES: print(f" {n:>7}", end="")
    print()
    for i, src in enumerate(STAGE_NAMES):
        print(f"{src:>6}", end="")
        for j in range(N_CLASSES): print(f" {M_mean[i,j]:>7.4f}", end="")
        print()

    print("\nTransition Matrix Std Dev (teacher consistency):")
    print(f"{'':>6}", end="")
    for n in STAGE_NAMES: print(f" {n:>7}", end="")
    print()
    for i, src in enumerate(STAGE_NAMES):
        print(f"{src:>6}", end="")
        for j in range(N_CLASSES): print(f" {M_std[i,j]:>7.4f}", end="")
        print(f"   (n={M_n[i]})")

    # Report per-subject pair counts (sanity check for the split step)
    unique_subj, counts = np.unique(dataset['subject_id'], return_counts=True)
    print(f"\nPairs per subject:")
    for s, c in zip(unique_subj, counts):
        print(f"  {s}: {c:,} pairs")

    print(f"\nDataset: {N_pairs:,} consecutive epoch pairs across "
          f"{len(unique_subj)} subjects")
    
    


if __name__ == "__main__":
    main()