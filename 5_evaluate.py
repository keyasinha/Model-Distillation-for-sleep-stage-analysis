import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings("ignore")

STAGE_NAMES = ["Wake", "N1", "N2", "N3", "REM"]
N_CLASSES   = 5

DARK_BG   = "#0f1117"
PANEL_BG  = "#1a1d27"
GRID_COL  = "#2a2d3a"
TEXT_COL  = "#e8eaf0"
TEXT_DIM  = "#8890a8"

COLOR_BASE = "#5b8dee"   
COLOR_PROP = "#e8643a"   
STAGE_COLORS = ["#6c8ebf", "#82b366", "#d6a920", "#ae4132", "#9b59b6"]

def set_dark_style():
    plt.rcParams.update({
        'figure.facecolor':  DARK_BG,
        'axes.facecolor':    PANEL_BG,
        'axes.edgecolor':    GRID_COL,
        'axes.labelcolor':   TEXT_COL,
        'axes.grid':         True,
        'grid.color':        GRID_COL,
        'grid.linewidth':    0.6,
        'text.color':        TEXT_COL,
        'xtick.color':       TEXT_DIM,
        'ytick.color':       TEXT_DIM,
        'legend.facecolor':  PANEL_BG,
        'legend.edgecolor':  GRID_COL,
        'font.family':       'sans-serif',
        'font.size':         10,
    })


def plot_training_curves(results, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Training Curves — Baseline KD vs Proposed TAD",
                 fontsize=15, fontweight='bold', y=0.98)

    metrics = [
        ('train_loss', 'Training Loss',    True),
        ('val_acc',    'Val Accuracy',     False),
        ('val_kappa',  'Val Cohen κ',      False),
        ('val_f1',     'Val Macro F1',     False),
    ]

    for ax, (key, title, log_scale) in zip(axes.flat, metrics):
        for name, color, label in [
            ('baseline_kd', COLOR_BASE, 'Baseline KD (γ=0)'),
            ('proposed_tad', COLOR_PROP, 'Proposed TAD (γ=0.2)'),
        ]:
            if name not in results:
                continue
            y = results[name]['history'][key]
            x = range(1, len(y) + 1)
            ax.plot(x, y, color=color, linewidth=2, label=label)
            # Smooth line
            if len(y) > 5:
                from scipy.ndimage import gaussian_filter1d
                y_sm = gaussian_filter1d(y, sigma=2)
                ax.plot(x, y_sm, color=color, linewidth=1, alpha=0.4, linestyle='--')

        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel("Epoch")
        ax.legend(fontsize=9)
        if log_scale:
            ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
    plt.close()
    print(f"Saved: {out_path}")

def plot_confusion_matrices(results, out_path):
    models = [k for k in ['baseline_kd', 'proposed_tad'] if k in results]
    fig, axes = plt.subplots(1, len(models), figsize=(7 * len(models), 6))
    if len(models) == 1:
        axes = [axes]
    fig.suptitle("Confusion Matrices (Normalised by True Label)",
                 fontsize=14, fontweight='bold')

    cmap = LinearSegmentedColormap.from_list(
        'dark_heat', [PANEL_BG, '#2d4a8a', '#5b8dee', '#e8643a', '#f5c842'])

    for ax, name in zip(axes, models):
        y_true = np.array(results[name]['true_labels'])
        y_pred = np.array(results[name]['predictions'])
        cm = confusion_matrix(y_true, y_pred, labels=range(N_CLASSES),
                              normalize='true')

        im = ax.imshow(cm, cmap=cmap, vmin=0, vmax=1, aspect='auto')
        ax.set_xticks(range(N_CLASSES))
        ax.set_yticks(range(N_CLASSES))
        ax.set_xticklabels(STAGE_NAMES, fontsize=11)
        ax.set_yticklabels(STAGE_NAMES, fontsize=11)
        ax.set_xlabel("Predicted", fontsize=12)
        ax.set_ylabel("True",      fontsize=12)

        label = "Baseline KD (γ=0)" if name == 'baseline_kd' else "Proposed TAD (γ=0.2)"
        kappa = results[name]['kappa']
        acc   = results[name]['accuracy']
        ax.set_title(f"{label}\nAcc={acc:.3f}  κ={kappa:.3f}",
                     fontsize=12, fontweight='bold')

        for i in range(N_CLASSES):
            for j in range(N_CLASSES):
                val = cm[i, j]
                col = 'black' if val > 0.6 else TEXT_COL
                ax.text(j, i, f"{val:.2f}", ha='center', va='center',
                        fontsize=10, color=col, fontweight='bold')

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
    plt.close()
    print(f"Saved: {out_path}")

def plot_per_stage_f1(results, out_path):
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle("Per-Stage F1 Score — Baseline vs Proposed",
                 fontsize=14, fontweight='bold')

    x = np.arange(N_CLASSES)
    width = 0.35

    for offset, (name, color, label) in enumerate([
        ('baseline_kd',  COLOR_BASE, 'Baseline KD (γ=0)'),
        ('proposed_tad', COLOR_PROP, 'Proposed TAD (γ=0.2)'),
    ]):
        if name not in results:
            continue
        f1s = results[name]['stage_f1']
        bars = ax.bar(x + offset * width - width/2, f1s, width,
                      label=label, color=color, alpha=0.85, zorder=3)
        for bar, val in zip(bars, f1s):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha='center', va='bottom', fontsize=9,
                    color=TEXT_COL, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(STAGE_NAMES, fontsize=12)
    ax.set_ylabel("F1 Score", fontsize=12)
    ax.set_ylim(0, 1.08)
    ax.legend(fontsize=10)
    ax.set_title("Note: N1 is hardest — highest improvement expected there",
                 fontsize=10, color=TEXT_DIM)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
    plt.close()
    print(f"Saved: {out_path}")

def compute_transition_accuracy(y_true, y_pred):
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    N = len(y_true) - 1
    correct = 0
    per_stage = {i: {'correct': 0, 'total': 0} for i in range(N_CLASSES)}

    for t in range(N):
        true_trans  = (y_true[t], y_true[t+1])
        pred_trans  = (y_pred[t], y_pred[t+1])
        match = (true_trans == pred_trans)
        if match:
            correct += 1
        s_src = y_true[t]
        per_stage[s_src]['total']   += 1
        per_stage[s_src]['correct'] += int(match)

    overall = correct / N
    per_stage_acc = {
        i: (per_stage[i]['correct'] / per_stage[i]['total']
            if per_stage[i]['total'] > 0 else 0.0)
        for i in range(N_CLASSES)
    }
    return overall, per_stage_acc


def plot_transition_accuracy(results, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Transition Accuracy — Novel Metric\n"
                 "% of epoch-pairs where predicted transition matches true transition",
                 fontsize=13, fontweight='bold')

    # Overall bar
    ax = axes[0]
    overall_vals, names_labels = [], []
    for name, label, color in [
        ('baseline_kd',  'Baseline KD', COLOR_BASE),
        ('proposed_tad', 'Proposed TAD', COLOR_PROP),
    ]:
        if name not in results:
            continue
        ov, _ = compute_transition_accuracy(
            results[name]['true_labels'], results[name]['predictions'])
        bar = ax.bar(label, ov, color=color, alpha=0.85, zorder=3, width=0.4)
        ax.text(bar[0].get_x() + bar[0].get_width()/2,
                ov + 0.005, f"{ov:.4f}",
                ha='center', va='bottom', fontsize=13,
                color=TEXT_COL, fontweight='bold')

    ax.set_ylabel("Transition Accuracy", fontsize=12)
    ax.set_ylim(0, 1.0)
    ax.set_title("Overall", fontsize=12)

    # Per-stage grouped bar
    ax2 = axes[1]
    x = np.arange(N_CLASSES)
    width = 0.35
    for offset, (name, color, label) in enumerate([
        ('baseline_kd',  COLOR_BASE, 'Baseline KD'),
        ('proposed_tad', COLOR_PROP, 'Proposed TAD'),
    ]):
        if name not in results:
            continue
        _, per_stage = compute_transition_accuracy(
            results[name]['true_labels'], results[name]['predictions'])
        vals = [per_stage[i] for i in range(N_CLASSES)]
        bars = ax2.bar(x + offset * width - width/2, vals, width,
                       color=color, alpha=0.85, label=label, zorder=3)
        for bar, val in zip(bars, vals):
            ax2.text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.01, f"{val:.2f}",
                     ha='center', va='bottom', fontsize=8, color=TEXT_COL)

    ax2.set_xticks(x)
    ax2.set_xticklabels(STAGE_NAMES, fontsize=11)
    ax2.set_ylabel("Per-Stage Transition Accuracy", fontsize=11)
    ax2.set_ylim(0, 1.08)
    ax2.legend(fontsize=10)
    ax2.set_title("By Source Stage", fontsize=12)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
    plt.close()
    print(f"Saved: {out_path}")

def plot_model_comparison(results, teacher_params, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Model Efficiency Comparison", fontsize=14, fontweight='bold')
    ax = axes[0]
    model_names  = ["Teacher\n(TinySleepNet)", "Student\nBaseline KD", "Student\nProposed TAD"]
    param_counts = [
        teacher_params,
        results.get('baseline_kd',  {}).get('params', 0),
        results.get('proposed_tad', {}).get('params', 0),
    ]
    colors = ["#888", COLOR_BASE, COLOR_PROP]
    bars = ax.bar(model_names, [p/1e3 for p in param_counts],
                  color=colors, alpha=0.85, zorder=3)
    for bar, val in zip(bars, param_counts):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.5,
                f"{val:,}", ha='center', va='bottom', fontsize=9,
                color=TEXT_COL, fontweight='bold')
    ax.set_ylabel("Parameters (K)", fontsize=11)
    ax.set_title("Parameter Count", fontsize=12)

    # Compression ratio annotation
    if param_counts[0] > 0 and param_counts[1] > 0:
        ratio = param_counts[0] / param_counts[1]
        ax.text(0.5, 0.9, f"Compression: {ratio:.0f}×",
                transform=ax.transAxes, ha='center', fontsize=11,
                color='#f5c842', fontweight='bold')

    # --- Inference speed ---
    ax2 = axes[1]
    speed_names  = ["Student\nBaseline KD", "Student\nProposed TAD"]
    speed_colors = [COLOR_BASE, COLOR_PROP]
    for i, (name, color) in enumerate([
        ('baseline_kd', COLOR_BASE), ('proposed_tad', COLOR_PROP)
    ]):
        if name not in results:
            continue
        spd = results[name]['speed_ms']
        bar = ax2.bar(speed_names[i], spd['mean_ms'], color=color,
                      alpha=0.85, zorder=3, yerr=spd['std_ms'],
                      error_kw={'color': TEXT_DIM, 'capsize': 5})
        ax2.text(bar[0].get_x() + bar[0].get_width()/2,
                 spd['mean_ms'] + spd['std_ms'] + 0.01,
                 f"{spd['mean_ms']:.3f} ms", ha='center', va='bottom',
                 fontsize=10, color=TEXT_COL, fontweight='bold')

    ax2.set_ylabel("Inference Time (ms/epoch)", fontsize=11)
    ax2.set_title("Inference Speed\n(lower = better)", fontsize=12)

    # --- Summary metrics bar ---
    ax3 = axes[2]
    metric_names = ["Accuracy", "Cohen κ", "Macro F1"]
    x = np.arange(len(metric_names))
    width = 0.35
    for offset, (name, color, label) in enumerate([
        ('baseline_kd',  COLOR_BASE, 'Baseline KD'),
        ('proposed_tad', COLOR_PROP, 'Proposed TAD'),
    ]):
        if name not in results:
            continue
        vals = [
            results[name]['accuracy'],
            results[name]['kappa'],
            results[name]['macro_f1'],
        ]
        bars = ax3.bar(x + offset * width - width/2, vals, width,
                       color=color, alpha=0.85, label=label, zorder=3)
        for bar, val in zip(bars, vals):
            ax3.text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.005, f"{val:.3f}",
                     ha='center', va='bottom', fontsize=9, color=TEXT_COL)

    ax3.set_xticks(x)
    ax3.set_xticklabels(metric_names, fontsize=11)
    ax3.set_ylim(0, 1.1)
    ax3.set_title("Overall Metrics", fontsize=12)
    ax3.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
    plt.close()
    print(f"Saved: {out_path}")


# -------------------------------------------------------------------
# 6. Loss component breakdown
# -------------------------------------------------------------------
def plot_loss_breakdown(results, out_path):
    if 'proposed_tad' not in results:
        return
    hist = results['proposed_tad']['history']
    epochs = range(1, len(hist['loss_ce']) + 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle("Loss Component Breakdown — Proposed TAD",
                 fontsize=13, fontweight='bold')

    ax.plot(epochs, hist['loss_ce'],    color="#82b366", linewidth=2, label="L_CE (hard labels)")
    ax.plot(epochs, hist['loss_kd'],    color="#5b8dee", linewidth=2, label="L_KD (soft labels)")
    ax.plot(epochs, hist['loss_trans'], color="#e8643a", linewidth=2, label="L_trans (transitions)")
    ax.plot(epochs, hist['train_loss'], color="white",   linewidth=1.5,
            linestyle='--', alpha=0.6, label="L_total")

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.legend(fontsize=10)
    ax.set_title("Each component's contribution over training", fontsize=11,
                 color=TEXT_DIM)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
    plt.close()
    print(f"Saved: {out_path}")


# -------------------------------------------------------------------
# 7. Summary table as PNG
# -------------------------------------------------------------------
def plot_summary_table(results, teacher_params, out_path):
    fig, ax = plt.subplots(figsize=(13, 4))
    fig.suptitle("Summary: Teacher vs Baseline Student vs Proposed Student",
                 fontsize=13, fontweight='bold', y=1.01)
    ax.axis('off')

    columns = ["Model", "Params", "Compression", "Accuracy", "Cohen κ",
               "Macro F1", "Inf. Time (ms)", "N1 F1", "REM F1"]

    def fmt_speed(name):
        if name not in results:
            return "—"
        return f"{results[name]['speed_ms']['mean_ms']:.3f}"

    def fmt_stage_f1(name, idx):
        if name not in results:
            return "—"
        return f"{results[name]['stage_f1'][idx]:.3f}"

    def compression(name):
        if name not in results or teacher_params == 0:
            return "—"
        return f"{teacher_params / results[name]['params']:.0f}×"

    rows = [
        ["Teacher (TinySleepNet)",
         f"{teacher_params:,}", "1×",
         "See step 2", "—", "—", "~5–15", "—", "—"],
        ["Student — Baseline KD (γ=0)",
         f"{results.get('baseline_kd',{}).get('params',0):,}",
         compression('baseline_kd'),
         f"{results.get('baseline_kd',{}).get('accuracy',0):.4f}",
         f"{results.get('baseline_kd',{}).get('kappa',0):.4f}",
         f"{results.get('baseline_kd',{}).get('macro_f1',0):.4f}",
         fmt_speed('baseline_kd'),
         fmt_stage_f1('baseline_kd', 1),
         fmt_stage_f1('baseline_kd', 4)],
        ["Student — Proposed TAD (γ=0.2)",
         f"{results.get('proposed_tad',{}).get('params',0):,}",
         compression('proposed_tad'),
         f"{results.get('proposed_tad',{}).get('accuracy',0):.4f}",
         f"{results.get('proposed_tad',{}).get('kappa',0):.4f}",
         f"{results.get('proposed_tad',{}).get('macro_f1',0):.4f}",
         fmt_speed('proposed_tad'),
         fmt_stage_f1('proposed_tad', 1),
         fmt_stage_f1('proposed_tad', 4)],
    ]

    row_colors = [
        ["#2a2d3a"] * len(columns),
        [COLOR_BASE + "30"] * len(columns),
        [COLOR_PROP + "30"] * len(columns),
    ]

    table = ax.table(
        cellText=rows, colLabels=columns,
        cellLoc='center', loc='center',
        cellColours=row_colors,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1, 2.2)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(GRID_COL)
        cell.set_text_props(color=TEXT_COL)
        if row == 0:
            cell.set_facecolor("#1f2235")
            cell.set_text_props(color=TEXT_COL, fontweight='bold')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
    plt.close()
    print(f"Saved: {out_path}")


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
def main():
    set_dark_style()
    os.makedirs("plots", exist_ok=True)

    results_path = "outputs/training_results.json"
    if not os.path.exists(results_path):
        print("Run 4_train_student.py first.")
        return

    with open(results_path) as f:
        results = json.load(f)

    # TinySleepNet param count — computed from architecture
    # (loaded from step 2 if available, otherwise hardcoded estimate)
    try:
        import sys
        sys.path.insert(0, ".")
        from importlib import import_module
        teacher_mod = import_module("2_teacher_inference".replace("-","_"))
        teacher_params = teacher_mod.TinySleepNet().count_params()
    except Exception:
        teacher_params = 1_280_000   # approximate TinySleepNet param count

    print("\nGenerating all evaluation plots...\n")

    plot_training_curves(
        results, "plots/training_curves.png")

    plot_confusion_matrices(
        results, "plots/confusion_matrices.png")

    plot_per_stage_f1(
        results, "plots/per_stage_f1.png")

    plot_transition_accuracy(
        results, "plots/transition_accuracy.png")

    plot_model_comparison(
        results, teacher_params, "plots/model_comparison.png")

    plot_loss_breakdown(
        results, "plots/loss_breakdown.png")

    plot_summary_table(
        results, teacher_params, "plots/summary_table.png")

    # Print final comparison
    print("\n" + "="*65)
    print("FINAL RESULTS SUMMARY")
    print("="*65)
    for name, label in [
        ('baseline_kd',  'Baseline KD  (γ=0)  '),
        ('proposed_tad', 'Proposed TAD (γ=0.2)'),
    ]:
        if name not in results:
            continue
        r = results[name]
        ov, _ = compute_transition_accuracy_from_lists(
            r['true_labels'], r['predictions'])
        print(f"\n{label}")
        print(f"  Params       : {r['params']:,}")
        print(f"  Accuracy     : {r['accuracy']:.4f}")
        print(f"  Cohen κ      : {r['kappa']:.4f}")
        print(f"  Macro F1     : {r['macro_f1']:.4f}")
        print(f"  Trans. Acc.  : {ov:.4f}")
        print(f"  Inf. time    : {r['speed_ms']['mean_ms']:.3f} ms/epoch")
        stage_f1 = r['stage_f1']
        print(f"  Per-stage F1 : " +
              " | ".join([f"{s}={v:.3f}" for s, v
                          in zip(["W","N1","N2","N3","REM"], stage_f1)]))

    print("\n" + "="*65)
    print("All plots saved to plots/")


def compute_transition_accuracy_from_lists(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    N = len(y_true) - 1
    correct = sum(
        1 for t in range(N)
        if (y_true[t] == y_true[t] and
            y_pred[t] == y_pred[t] and
            (y_true[t], y_true[t+1]) == (y_pred[t], y_pred[t+1]))
    )
    per_stage = {}
    for i in range(N_CLASSES):
        idxs = [t for t in range(N) if y_true[t] == i]
        if not idxs:
            per_stage[i] = 0.0
            continue
        per_stage[i] = sum(
            1 for t in idxs
            if (y_true[t], y_true[t+1]) == (y_pred[t], y_pred[t+1])
        ) / len(idxs)
    return correct / N, per_stage


if __name__ == "__main__":
    main()