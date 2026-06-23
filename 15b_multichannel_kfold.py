"""
  - eeg      : EEG Fpz-Cz only
  - eog      : EOG horizontal only
  - eeg_eog  : EEG Fpz-Cz + EOG horizontal
  - eog_emg  : EOG horizontal + EMG submental
"""

import os
import glob
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import cohen_kappa_score, f1_score, accuracy_score
from sklearn.model_selection import KFold
import mne
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

mne.set_log_level("WARNING")

from split_utils import verify_no_leakage

CHANNEL_CONFIGS = {
    "eeg": {
        "channels":    ["EEG Fpz-Cz"],
        "n_channels":  1,
        "label":       "EEG only",
        "description": "Single frontal EEG channel (Fpz-Cz)",
        "color":       "3B82F6",
    },
    "eog": {
        "channels":    ["EOG horizontal"],
        "n_channels":  1,
        "label":       "EOG only",
        "description": "Horizontal eye movement channel",
        "color":       "10B981",
    },
    "eeg_eog": {
        "channels":    ["EEG Fpz-Cz", "EOG horizontal"],
        "n_channels":  2,
        "label":       "EEG + EOG",
        "description": "Frontal EEG + horizontal EOG",
        "color":       "F59E0B",
    },
    "eog_emg": {
        "channels":    ["EOG horizontal", "EMG submental"],
        "n_channels":  2,
        "label":       "EOG + EMG",
        "description": "Eye movement + chin muscle activity",
        "color":       "EF4444",
    },
}

STAGE_NAMES = ["Wake", "N1", "N2", "N3", "REM"]
N_CLASSES   = 5
EPOCH_SEC   = 30
SAMPLE_RATE = 100
EPOCH_LEN   = EPOCH_SEC * SAMPLE_RATE
SEQ_LEN     = 20

ANNOTATION_MAP = {
    "Sleep stage W": 0, "Sleep stage 1": 1, "Sleep stage 2": 2,
    "Sleep stage 3": 3, "Sleep stage 4": 3, "Sleep stage R": 4,
    "Movement time": 0,
}


def extract_channels(psg_path, channels):
    raw = mne.io.read_raw_edf(psg_path, preload=True, verbose=False)
    available = raw.ch_names

    name_map = {
        "EEG Fpz-Cz":    ["EEG Fpz-Cz", "EEG Fpz-Cz."],
        "EOG horizontal": ["EOG horizontal", "EOG horizontal.", "EOG ROC-LOC"],
        "EMG submental":  ["EMG submental", "EMG submental.", "EMG chin"],
    }
    resolved = []
    for ch in channels:
        aliases = name_map.get(ch, [ch])
        matched = next((a for a in aliases if a in available), None)
        if matched is None:
            print(f" Channel '{ch}' not found. Available: {available}")
            return None, None
        resolved.append(matched)

    raw.pick_channels(resolved)
    data, _ = raw[:]
    return data, resolved


def prepare_subject(psg_path, hyp_path, channels):
    data, resolved = extract_channels(psg_path, channels)
    if data is None:
        return None, None

    n_ch = data.shape[0]

    ann = mne.read_annotations(hyp_path)
    n_samples = data.shape[1]
    n_epochs  = n_samples // EPOCH_LEN
    labels    = np.full(n_epochs, -1, dtype=np.int8)

    for onset, duration, desc in zip(ann.onset, ann.duration, ann.description):
        label = ANNOTATION_MAP.get(desc, -1)
        if label == -1: continue
        start_epoch = int(onset // EPOCH_SEC)
        for e in range(int(duration // EPOCH_SEC)):
            idx = start_epoch + e
            if 0 <= idx < n_epochs:
                labels[idx] = label

    epochs_list, valid_labels = [], []
    for i in range(n_epochs):
        if labels[i] == -1: continue
        seg = data[:, i * EPOCH_LEN : (i+1) * EPOCH_LEN]
        if seg.shape[1] == EPOCH_LEN:
            epochs_list.append(seg)
            valid_labels.append(labels[i])

    X = np.array(epochs_list, dtype=np.float32)
    y = np.array(valid_labels, dtype=np.int64)

    for c in range(n_ch):
        mu  = X[:, c, :].mean()
        std = X[:, c, :].std() + 1e-8
        X[:, c, :] = (X[:, c, :] - mu) / std

    return X, y


def prepare_all_subjects(raw_dir, out_dir, channels, config_name):
    os.makedirs(out_dir, exist_ok=True)
    psg_files = sorted(glob.glob(os.path.join(raw_dir, "*PSG.edf")))

    if not psg_files:
        print(f"No PSG files found in {raw_dir}")
        return []

    processed = []
    for psg_path in tqdm(psg_files, desc=f"Preparing {config_name}"):
        subj_id = os.path.basename(psg_path)[:6]
        out_path = os.path.join(out_dir, f"{subj_id}.npz")
        if os.path.exists(out_path):
            processed.append(subj_id)
            continue

        hyp_path = None
        for suffix in ["EC", "EH"]:
            candidate = psg_path.replace("E0-PSG.edf", f"{suffix}-Hypnogram.edf")
            if os.path.exists(candidate):
                hyp_path = candidate; break

        if hyp_path is None:
            tqdm.write(f" skipped - No hypnogram for {subj_id}")
            continue

        X, y = prepare_subject(psg_path, hyp_path, channels)
        if X is None:
            tqdm.write(f"  skipped - Missing channels for {subj_id}")
            continue

        np.savez(out_path, x=X, y=y, subject=subj_id)
        processed.append(subj_id)
        tqdm.write(f"  {subj_id}: {len(y)} epochs, shape {X.shape}")

    return processed


class TinySleepNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=5, fs=100, seq_len=20, dropout=0.5):
        super().__init__()
        self.seq_len = seq_len
        lf = fs // 2
        sf = fs // 4

        self.cnn_large = nn.Sequential(
            nn.Conv1d(n_channels, 128, kernel_size=lf, stride=lf//2, padding=lf//2, bias=False),
            nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(8, stride=8), nn.Dropout(dropout),
            nn.Conv1d(128, 128, kernel_size=8, padding=4, bias=False),
            nn.Conv1d(128, 128, kernel_size=8, padding=4, bias=False),
            nn.Conv1d(128, 128, kernel_size=8, padding=4, bias=False),
            nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(4, stride=4),
        )
        self.cnn_small = nn.Sequential(
            nn.Conv1d(n_channels, 128, kernel_size=sf, stride=sf//2, padding=sf//2, bias=False),
            nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(4, stride=4), nn.Dropout(dropout),
            nn.Conv1d(128, 128, kernel_size=6, padding=3, bias=False),
            nn.Conv1d(128, 128, kernel_size=6, padding=3, bias=False),
            nn.Conv1d(128, 128, kernel_size=6, padding=3, bias=False),
            nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2, stride=2),
        )
        self.cnn_drop = nn.Dropout(dropout)

        with torch.no_grad():
            dummy = torch.zeros(1, n_channels, EPOCH_LEN)
            l_out = self.cnn_large(dummy).flatten(1)
            s_out = self.cnn_small(dummy).flatten(1)
            cnn_dim = l_out.shape[1] + s_out.shape[1]

        self.lstm = nn.LSTM(cnn_dim, 128, num_layers=2, batch_first=True, dropout=dropout)
        self.lstm_drop = nn.Dropout(dropout)
        self.fc = nn.Linear(128, n_classes)

    def forward_cnn(self, x):
        l = self.cnn_large(x).flatten(1)
        s = self.cnn_small(x).flatten(1)
        return self.cnn_drop(torch.cat([l, s], dim=1))

    def forward(self, x, h=None):
        B, T, C, L = x.shape
        feats = self.forward_cnn(x.reshape(B*T, C, L)).reshape(B, T, -1)
        out, h_new = self.lstm(feats, h)
        out = self.lstm_drop(out)
        return self.fc(out), h_new

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class TeacherSeqDataset(Dataset):
    def __init__(self, npz_files, seq_len=SEQ_LEN):
        self.seq_len = seq_len
        self.sequences = []
        for f in npz_files:
            d = np.load(f)
            X, y = d['x'], d['y']
            n_seq = len(X) // seq_len
            for i in range(n_seq):
                s, e = i*seq_len, (i+1)*seq_len
                self.sequences.append((X[s:e], y[s:e]))

    def __len__(self): return len(self.sequences)

    def __getitem__(self, idx):
        X, y = self.sequences[idx]
        return torch.FloatTensor(X), torch.LongTensor(y)


def train_teacher(model, train_files, val_files, device, epochs, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    train_ds = TeacherSeqDataset(train_files)
    val_ds   = TeacherSeqDataset(val_files)
    train_dl = DataLoader(train_ds, batch_size=15, shuffle=True,  num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=15, shuffle=False, num_workers=0)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    best_f1 = -1.0

    print(f"  Teacher params: {model.count_params():,}")

    for ep in range(1, epochs+1):
        model.train()
        for X_seq, y_seq in tqdm(train_dl, desc=f"  Ep {ep}/{epochs}", leave=False):
            X_seq, y_seq = X_seq.to(device), y_seq.to(device)
            optimizer.zero_grad()
            logits, _ = model(X_seq)
            loss = F.cross_entropy(logits.reshape(-1, N_CLASSES), y_seq.reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        preds_all, true_all = [], []
        with torch.no_grad():
            for X_seq, y_seq in val_dl:
                X_seq = X_seq.to(device)
                logits, _ = model(X_seq)
                preds_all.extend(logits.argmax(-1).reshape(-1).cpu().numpy())
                true_all.extend(y_seq.reshape(-1).numpy())

        preds_all = np.array(preds_all); true_all = np.array(true_all)
        acc      = accuracy_score(true_all, preds_all)
        kappa    = cohen_kappa_score(true_all, preds_all)
        macro_f1 = f1_score(true_all, preds_all, average='macro', zero_division=0)
        print(f"  Ep {ep:3d} | acc={acc:.4f} κ={kappa:.4f} F1={macro_f1:.4f}")

        if macro_f1 > best_f1:
            best_f1 = macro_f1
            torch.save(model.state_dict(), save_path)

    model.load_state_dict(torch.load(save_path, map_location=device))
    return model


def extract_soft_labels(model, npz_files, device, temperature, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    results = {}
    for f in tqdm(npz_files, desc="  Extracting soft labels"):
        subj = os.path.basename(f).replace(".npz","")
        out_path = os.path.join(out_dir, f"{subj}_soft.npy")
        if os.path.exists(out_path):
            results[subj] = np.load(out_path)
            continue

        d = np.load(f)
        X, y = d['x'], d['y']
        N = len(X)
        soft_labels = np.zeros((N, N_CLASSES), dtype=np.float32)

        with torch.no_grad():
            idx = 0
            while idx < N:
                end = min(idx + SEQ_LEN, N)
                chunk = X[idx:end]
                if len(chunk) < SEQ_LEN:
                    n_ch = chunk.shape[1]
                    pad = np.zeros((SEQ_LEN-len(chunk), n_ch, EPOCH_LEN), dtype=np.float32)
                    chunk = np.concatenate([chunk, pad], axis=0)
                x_t = torch.FloatTensor(chunk).unsqueeze(0).to(device)
                logits, _ = model(x_t)
                probs = F.softmax(logits[0] / temperature, dim=-1).cpu().numpy()
                actual = end - idx
                soft_labels[idx:end] = probs[:actual]
                idx = end

        np.save(out_path, soft_labels)
        results[subj] = soft_labels
    return results


def build_transition_dataset(npz_files, soft_label_dir, out_path):
    if os.path.exists(out_path):
        print(f"path: {out_path}")
        return

    all_x_t, all_x_t1, all_p_t, all_p_t1 = [], [], [], []
    all_y_t, all_s_t, all_w_t, all_subj  = [], [], [], []

    for f in npz_files:
        subj = os.path.basename(f).replace(".npz","")
        soft_path = os.path.join(soft_label_dir, f"{subj}_soft.npy")
        if not os.path.exists(soft_path): continue

        d = np.load(f)
        X, y = d['x'], d['y']
        soft = np.load(soft_path)
        min_len = min(len(X), len(soft))
        X, y, soft = X[:min_len], y[:min_len], soft[:min_len]

        for t in range(len(X) - 1):
            all_x_t.append(X[t])
            all_x_t1.append(X[t+1])
            all_p_t.append(soft[t])
            all_p_t1.append(soft[t+1])
            all_y_t.append(int(y[t]))
            all_s_t.append(int(np.argmax(soft[t])))
            all_w_t.append(float(np.max(soft[t])))
            all_subj.append(subj)

    np.savez_compressed(out_path,
        x_t        = np.array(all_x_t,  dtype=np.float32),
        x_t1       = np.array(all_x_t1, dtype=np.float32),
        p_t        = np.array(all_p_t,  dtype=np.float32),
        p_t1       = np.array(all_p_t1, dtype=np.float32),
        y_t        = np.array(all_y_t,  dtype=np.int64),
        s_t        = np.array(all_s_t,  dtype=np.int64),
        w_t        = np.array(all_w_t,  dtype=np.float32),
        subject_id = np.array(all_subj, dtype="<U10"),
    )
    print(f"  Saved {len(all_y_t):,} pairs -> {out_path}")


class SleepStudentNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=5, gru_hidden=64, dropout=0.3):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(n_channels, 16, kernel_size=50, stride=5, padding=25),
            nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=10, stride=2, padding=5),
            nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32), nn.ReLU(), nn.AdaptiveAvgPool1d(32),
        )
        self.dropout = nn.Dropout(dropout)
        self.gru     = nn.GRU(32, gru_hidden, batch_first=True)
        self.fc      = nn.Linear(gru_hidden, n_classes)

    def forward(self, x, h=None):
        if x.dim() == 2: x = x.unsqueeze(1)
        feat = self.cnn(x).permute(0, 2, 1)
        feat = self.dropout(feat)
        out, h_new = self.gru(feat, h)
        return self.fc(out[:, -1, :]), h_new

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class TransitionDataset(Dataset):
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


def compute_student_loss(student, batch, device, alpha, beta, gamma, tau=2.0):
    x_t  = batch['x_t'].to(device)
    x_t1 = batch['x_t1'].to(device)
    p_t  = batch['p_t'].to(device)
    p_t1 = batch['p_t1'].to(device)
    y_t  = batch['y_t'].to(device)

    logits_t,  h = student(x_t)
    logits_t1, _ = student(x_t1, h.detach())

    L_ce = F.cross_entropy(logits_t, y_t)

    L_kd = torch.tensor(0.0, device=device)
    if beta > 0:
        L_kd = F.kl_div(
            F.log_softmax(logits_t / tau, dim=-1),
            F.softmax(p_t / tau, dim=-1),
            reduction='batchmean') * (tau**2)

    L_trans = torch.tensor(0.0, device=device)
    if gamma > 0:
        kl_per  = F.kl_div(
            F.log_softmax(logits_t1 / tau, dim=-1),
            F.softmax(p_t1 / tau, dim=-1),
            reduction='none').sum(dim=-1)
        L_trans = kl_per.mean() * (tau**2)

    return alpha*L_ce + beta*L_kd + gamma*L_trans, L_ce.item(), L_kd.item(), L_trans.item()


def train_student_one_fold(cfg, train_dl, val_dl, device, n_channels, save_path, epochs=40):
    model = SleepStudentNet(n_channels=n_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    alpha, beta, gamma = cfg['alpha'], cfg['beta'], cfg['gamma']
    history = {'loss': [], 'ce': [], 'kd': [], 'trans': [],
               'acc': [], 'kappa': [], 'f1': [], 'stage_f1': []}
    best_f1 = -1.0
    best_metrics = {}

    for ep in range(1, epochs+1):
        model.train()
        ep_loss = {'total':0,'ce':0,'kd':0,'trans':0}; n=0
        for batch in tqdm(train_dl, desc=f"    Ep {ep:3d}/{epochs}", leave=False):
            optimizer.zero_grad()
            loss, ce, kd, tr = compute_student_loss(
                model, batch, device, alpha, beta, gamma)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ep_loss['total']+=loss.item(); ep_loss['ce']+=ce
            ep_loss['kd']+=kd; ep_loss['trans']+=tr; n+=1
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
        acc   = accuracy_score(true_all, preds_all)
        kappa = cohen_kappa_score(true_all, preds_all)
        f1    = f1_score(true_all, preds_all, average='macro', zero_division=0)
        sf1   = f1_score(true_all, preds_all, average=None,
                         labels=list(range(N_CLASSES)), zero_division=0)

        history['loss'].append(ep_loss['total'])
        history['ce'].append(ep_loss['ce'])
        history['kd'].append(ep_loss['kd'])
        history['trans'].append(ep_loss['trans'])
        history['acc'].append(acc)
        history['kappa'].append(kappa)
        history['f1'].append(f1)
        history['stage_f1'].append(sf1.tolist())

        print(f"    Ep {ep:3d} | ce={ep_loss['ce']:.4f} kd={ep_loss['kd']:.4f} "
              f"tr={ep_loss['trans']:.4f} | acc={acc:.4f} κ={kappa:.4f} "
              f"F1={f1:.4f} N1={sf1[1]:.3f}")

        if f1 > best_f1:
            best_f1 = f1
            best_metrics = {'accuracy':float(acc),'kappa':float(kappa),
                           'macro_f1':float(f1),'stage_f1':sf1.tolist(),'epoch':ep}
            torch.save(model.state_dict(), save_path)

    model.load_state_dict(torch.load(save_path, map_location=device))
    return model, history, best_metrics


def run_kfold_students(trans_path, subject_ids, n_channels, device,
                       student_configs, ckpt_dir, epochs, n_folds=10):
    """
    10-fold CV over subjects.
    Each fold: train on 9 folds of subjects, val on 1 fold.
    Returns per-student aggregated metrics (mean +- std over folds).
    """
    raw = np.load(trans_path)
    all_subject_ids = np.array(sorted(set(raw['subject_id'].tolist())))
    pair_subjects   = raw['subject_id']

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    # fold_results[sname] = list of best_metrics per fold
    fold_results = {sname: [] for sname in student_configs}

    for fold_idx, (train_subj_idx, val_subj_idx) in enumerate(kf.split(all_subject_ids)):
        train_subjs = set(all_subject_ids[train_subj_idx])
        val_subjs   = set(all_subject_ids[val_subj_idx])

        print(f"\n  Fold {fold_idx+1}/{n_folds} | "
              f"train={len(train_subjs)} subjects, val={len(val_subjs)} subjects")
        print(f"  Val subjects: {sorted(val_subjs)}")

        # get pair indices for this fold
        train_idx = np.where([s in train_subjs for s in pair_subjects])[0]
        val_idx   = np.where([s in val_subjs   for s in pair_subjects])[0]

        # leakage check
        leak = verify_no_leakage(raw['x_t'], raw['x_t1'], train_idx, val_idx)
        if leak > 0.001:
            print(f"  [SKIP] Leakage {leak:.2%} in fold {fold_idx+1}"); continue

        train_ds = TransitionDataset(trans_path, indices=train_idx)
        val_ds   = TransitionDataset(trans_path, indices=val_idx)
        train_dl = DataLoader(train_ds, batch_size=256, shuffle=True,  num_workers=0)
        val_dl   = DataLoader(val_ds,   batch_size=256, shuffle=False, num_workers=0)

        print(f"  {len(train_ds):,} train pairs | {len(val_ds):,} val pairs")

        for sname, scfg in student_configs.items():
            print(f"\n  [{sname}] fold {fold_idx+1}")
            save_path = os.path.join(ckpt_dir, f"student_{sname}_fold{fold_idx+1}.pt")
            _, history, best = train_student_one_fold(
                scfg, train_dl, val_dl, device,
                n_channels=n_channels,
                save_path=save_path,
                epochs=epochs)
            best['fold'] = fold_idx + 1
            fold_results[sname].append(best)
            print(f"  [{sname}] fold {fold_idx+1} best: "
                  f"acc={best['accuracy']:.4f} κ={best['kappa']:.4f} "
                  f"F1={best['macro_f1']:.4f} N1={best['stage_f1'][1]:.3f}")

    # aggregate across folds
    aggregated = {}
    for sname, folds in fold_results.items():
        if not folds:
            continue
        accs    = [f['accuracy']  for f in folds]
        kappas  = [f['kappa']     for f in folds]
        f1s     = [f['macro_f1']  for f in folds]
        n1s     = [f['stage_f1'][1] for f in folds]
        # per-stage mean over folds
        stage_f1_mean = np.mean([f['stage_f1'] for f in folds], axis=0).tolist()
        stage_f1_std  = np.std( [f['stage_f1'] for f in folds], axis=0).tolist()

        aggregated[sname] = {
            'fold_results': folds,
            'mean': {
                'accuracy':  float(np.mean(accs)),
                'kappa':     float(np.mean(kappas)),
                'macro_f1':  float(np.mean(f1s)),
                'n1_f1':     float(np.mean(n1s)),
                'stage_f1':  stage_f1_mean,
            },
            'std': {
                'accuracy':  float(np.std(accs)),
                'kappa':     float(np.std(kappas)),
                'macro_f1':  float(np.std(f1s)),
                'n1_f1':     float(np.std(n1s)),
                'stage_f1':  stage_f1_std,
            },
        }
        print(f"\n  [{sname}] {n_folds}-fold summary:")
        print(f"    acc  = {np.mean(accs):.4f} +- {np.std(accs):.4f}")
        print(f"    κ    = {np.mean(kappas):.4f} +- {np.std(kappas):.4f}")
        print(f"    F1   = {np.mean(f1s):.4f} +- {np.std(f1s):.4f}")
        print(f"    N1   = {np.mean(n1s):.4f} +- {np.std(n1s):.4f}")

    return aggregated


DARK_BG  = "#0f1117"
PANEL_BG = "#1a1d27"
GRID_COL = "#2a2d3a"
TEXT_COL = "#e8eaf0"
TEXT_DIM = "#8890a8"

ABLATION_COLORS = {"ce_only":"#5b8dee", "no_kd":"#82b366", "full_tad":"#e8643a"}
ABLATION_LABELS = {
    "ce_only":  "CE only  (alpha=1, beta=0, gamma=0)",
    "no_kd":    "CE + TAD  (beta=0, gamma=0.2)",
    "full_tad": "Full TAD  (alpha*CE + beta*KD + gamma*TAD)",
}


def dark_ax(ax, title=None):
    ax.set_facecolor(PANEL_BG)
    if title: ax.set_title(title, color=TEXT_COL, fontsize=10, fontweight='bold')
    ax.tick_params(colors=TEXT_DIM, labelsize=8)
    ax.grid(color=GRID_COL, linewidth=0.5)
    for spine in ax.spines.values(): spine.set_edgecolor(GRID_COL)


def plot_teacher_comparison(all_results, out_path):
    configs  = list(all_results.keys())
    colors   = [CHANNEL_CONFIGS[c]['color'] for c in configs]
    labels   = [CHANNEL_CONFIGS[c]['label'] for c in configs]

    metrics_keys   = ['accuracy', 'kappa', 'macro_f1']
    metrics_labels = ['Accuracy', 'Cohen κ', 'Macro F1']

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    fig.patch.set_facecolor(DARK_BG)
    fig.suptitle("Teacher Model Comparison - Channel Configurations",
                 color=TEXT_COL, fontsize=14, fontweight='bold')

    for ax, mk, ml in zip(axes[0][:3], metrics_keys, metrics_labels):
        dark_ax(ax, ml)
        vals = [all_results[c]['teacher'][mk] for c in configs]
        bars = ax.bar(labels, vals, color=[f"#{x}" for x in colors], alpha=0.85, zorder=3)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, v+0.003, f"{v:.4f}",
                    ha='center', va='bottom', fontsize=8, color=TEXT_COL, fontweight='bold')
        ax.set_ylim(min(vals)-0.05, 1.0)
        ax.set_xticklabels(labels, rotation=15, ha='right', color=TEXT_DIM, fontsize=8)

    ax_sf = axes[0][3]
    dark_ax(ax_sf, "Per-Stage F1 by Channel Config")
    x = np.arange(N_CLASSES)
    bar_w = 0.18
    for i, c in enumerate(configs):
        sf1 = all_results[c]['teacher']['stage_f1']
        ax_sf.bar(x + (i - len(configs)/2 + 0.5)*bar_w, sf1, bar_w,
                  color=f"#{CHANNEL_CONFIGS[c]['color']}", alpha=0.85, label=labels[i])
    ax_sf.set_xticks(x)
    ax_sf.set_xticklabels(STAGE_NAMES, color=TEXT_COL, fontsize=9)
    ax_sf.legend(fontsize=7, facecolor=PANEL_BG, labelcolor=TEXT_COL)

    for i, (c, ax) in enumerate(zip(configs, axes[1])):
        dark_ax(ax, f"{CHANNEL_CONFIGS[c]['label']} - Per-Stage F1")
        sf1 = all_results[c]['teacher']['stage_f1']
        col = f"#{CHANNEL_CONFIGS[c]['color']}"
        bars = ax.bar(STAGE_NAMES, sf1, color=col, alpha=0.85, zorder=3)
        for bar, v in zip(bars, sf1):
            ax.text(bar.get_x()+bar.get_width()/2, v+0.01, f"{v:.3f}",
                    ha='center', va='bottom', fontsize=8, color=TEXT_COL)
        ax.set_ylim(0, 1.1)
        ax.set_xticklabels(STAGE_NAMES, color=TEXT_COL, fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
    plt.close()
    print(f"Saved: {out_path}")


def plot_student_kfold(config_name, config_results, out_path):
    # bar chart with error bars showing mean +- std over folds
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    fig.patch.set_facecolor(DARK_BG)
    label = CHANNEL_CONFIGS[config_name]['label']
    fig.suptitle(f"Student 10-Fold CV - {label}",
                 color=TEXT_COL, fontsize=13, fontweight='bold')

    metric_keys   = ['accuracy', 'kappa', 'macro_f1', 'n1_f1']
    metric_labels = ['Accuracy', 'Cohen κ', 'Macro F1', 'N1 F1']
    ablations     = ['ce_only', 'no_kd', 'full_tad']

    for ax, mk, ml in zip(axes, metric_keys, metric_labels):
        dark_ax(ax, ml)
        means = [config_results['students'][s]['mean'][mk] for s in ablations]
        stds  = [config_results['students'][s]['std'][mk]  for s in ablations]
        colors = [ABLATION_COLORS[s] for s in ablations]
        bars = ax.bar(ablations, means, color=colors, alpha=0.85, zorder=3)
        ax.errorbar(range(len(ablations)), means, yerr=stds,
                    fmt='none', color=TEXT_COL, capsize=5, linewidth=1.5, zorder=4)
        for bar, v, s in zip(bars, means, stds):
            ax.text(bar.get_x()+bar.get_width()/2, v+s+0.005,
                    f"{v:.3f}", ha='center', va='bottom',
                    fontsize=8, color=TEXT_COL, fontweight='bold')
        ax.set_ylim(max(0, min(means)-max(stds)-0.05), 1.0)
        ax.set_xticklabels(['CE only', 'No KD', 'Full TAD'],
                           rotation=15, ha='right', color=TEXT_DIM, fontsize=8)

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", default=list(CHANNEL_CONFIGS.keys()),
                        choices=list(CHANNEL_CONFIGS.keys()))
    parser.add_argument("--skip_teacher", action="store_true")
    parser.add_argument("--teacher_epochs", type=int, default=30)
    parser.add_argument("--student_epochs", type=int, default=40)
    parser.add_argument("--n_folds",        type=int, default=10)
    parser.add_argument("--raw_dir",  type=str, default="data/raw")
    parser.add_argument("--base_dir", type=str, default="experiments")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(args.base_dir, exist_ok=True)
    os.makedirs("plots", exist_ok=True)

    all_results = {}

    for config_name in args.configs:
        cfg = CHANNEL_CONFIGS[config_name]
        print(f"\nCONFIG: {config_name}  ({cfg['label']})  -  channels: {cfg['channels']}")

        data_dir     = os.path.join(args.base_dir, config_name, "data")
        soft_dir     = os.path.join(args.base_dir, config_name, "soft_labels")
        ckpt_dir     = os.path.join(args.base_dir, config_name, "checkpoints")
        trans_path   = os.path.join(args.base_dir, config_name, "transition_dataset.npz")
        results_path = os.path.join(args.base_dir, config_name, "results.json")
        for d in [data_dir, soft_dir, ckpt_dir]: os.makedirs(d, exist_ok=True)

        print(f"\n1. Preparing {cfg['label']} data")
        subj_ids = prepare_all_subjects(args.raw_dir, data_dir, cfg['channels'], config_name)
        if not subj_ids:
            print(f"  No subjects prepared for {config_name}. Skipping.")
            continue

        npz_files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
        print(f"  {len(npz_files)} subjects ready.")

        # teacher uses a simple 80/20 split (yeh sirf teacher training ke liye hai,
        # student ka proper evaluation 10-fold mein hoga)
        n_val = max(1, int(len(npz_files) * 0.2))
        rng = np.random.default_rng(42)
        perm = rng.permutation(len(npz_files))
        train_files = [npz_files[i] for i in perm[n_val:]]
        val_files   = [npz_files[i] for i in perm[:n_val]]

        teacher_ckpt        = os.path.join(ckpt_dir, "teacher.pt")
        teacher_metrics_path = os.path.join(args.base_dir, config_name, "teacher_metrics.json")

        print(f"\n2. Teacher model ({cfg['label']})")
        teacher = TinySleepNet(n_channels=cfg['n_channels']).to(device)

        if args.skip_teacher and os.path.exists(teacher_ckpt):
            print(f"  Loading existing checkpoint: {teacher_ckpt}")
            teacher.load_state_dict(torch.load(teacher_ckpt, map_location=device))
            with open(teacher_metrics_path) as f:
                teacher_metrics = json.load(f)
        else:
            teacher = train_teacher(
                teacher, train_files, val_files, device,
                epochs=args.teacher_epochs, save_path=teacher_ckpt)

            val_ds = TeacherSeqDataset(val_files)
            val_dl = DataLoader(val_ds, batch_size=15, shuffle=False, num_workers=0)
            teacher.eval()
            preds_all, true_all = [], []
            with torch.no_grad():
                for X_seq, y_seq in val_dl:
                    X_seq = X_seq.to(device)
                    logits, _ = teacher(X_seq)
                    preds_all.extend(logits.argmax(-1).reshape(-1).cpu().numpy())
                    true_all.extend(y_seq.reshape(-1).numpy())
            preds_all = np.array(preds_all); true_all = np.array(true_all)
            teacher_metrics = {
                'accuracy':  float(accuracy_score(true_all, preds_all)),
                'kappa':     float(cohen_kappa_score(true_all, preds_all)),
                'macro_f1':  float(f1_score(true_all, preds_all, average='macro', zero_division=0)),
                'stage_f1':  f1_score(true_all, preds_all, average=None,
                                      labels=list(range(N_CLASSES)), zero_division=0).tolist(),
                'params':    teacher.count_params(),
            }
            with open(teacher_metrics_path, 'w') as f:
                json.dump(teacher_metrics, f, indent=2)

        print(f"  Teacher: acc={teacher_metrics['accuracy']:.4f} "
              f"κ={teacher_metrics['kappa']:.4f} F1={teacher_metrics['macro_f1']:.4f}")

        # soft labels from all subjects (teacher pura dataset pe chalta hain,
        # folds ke andar leakage nahi hogi because soft labels sirf input hain)
        print(f"\n3. Extracting soft labels (τ=2.0)")
        extract_soft_labels(teacher, npz_files, device, temperature=2.0, out_dir=soft_dir)

        print(f"\n4. Building transition dataset")
        build_transition_dataset(npz_files, soft_dir, trans_path)

        print(f"\n5. Training students - {args.n_folds}-fold CV")
        student_configs = {
            'ce_only':  {'alpha': 1.0, 'beta': 0.0, 'gamma': 0.0},
            'no_kd':    {'alpha': 1.0, 'beta': 0.0, 'gamma': 0.2},
            'full_tad': {'alpha': 0.3, 'beta': 0.5, 'gamma': 0.2},
        }

        student_results = run_kfold_students(
            trans_path, subj_ids, cfg['n_channels'], device,
            student_configs, ckpt_dir,
            epochs=args.student_epochs, n_folds=args.n_folds)

        config_results = {
            'teacher':  teacher_metrics,
            'students': student_results,
        }
        with open(results_path, 'w') as f:
            json.dump(_serial(config_results), f, indent=2)

        all_results[config_name] = config_results

        plot_student_kfold(
            config_name, config_results,
            os.path.join("plots", f"kfold_{config_name}.png"))

    if len(all_results) > 1:
        plot_teacher_comparison(all_results, "plots/teacher_comparison.png")

    print("\nSummary:")
    print(f"{'Config':<12}{'Student':<12}{'Acc mean':>10}{'Acc std':>9}"
          f"{'κ mean':>9}{'κ std':>8}{'F1 mean':>9}{'F1 std':>8}"
          f"{'N1 mean':>9}{'N1 std':>8}")
    for c, res in all_results.items():
        for sname, sres in res['students'].items():
            m = sres['mean']; s = sres['std']
            print(f"{c:<12}{sname:<12}{m['accuracy']:>10.4f}{s['accuracy']:>9.4f}"
                  f"{m['kappa']:>9.4f}{s['kappa']:>8.4f}"
                  f"{m['macro_f1']:>9.4f}{s['macro_f1']:>8.4f}"
                  f"{m['n1_f1']:>9.4f}{s['n1_f1']:>8.4f}")

    with open("experiments/all_results.json", 'w') as f:
        json.dump(_serial(all_results), f, indent=2)


if __name__ == "__main__":
    main()